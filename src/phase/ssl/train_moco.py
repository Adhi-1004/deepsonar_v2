from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from phase.augment import NoiseBank, Pipeline
from phase.config import PretrainConfig
from phase.data.dataset import WindowDataset
from phase.data.manifest import resolve
from phase.features.frontend import LogMelFrontEnd
from phase.models.moco import MoCo
from phase.seed import worker_init_fn
from phase.ssl.demon_consistency import DemonConsistency
from phase.ssl.losses import info_nce
from phase.ssl.monitors import health, is_collapsed, knn_accuracy


def cosine_lr(epoch: int, total: int, warmup: int, base: float) -> float:
    if epoch < warmup:
        return base * (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * progress))


def build_loaders(
    config: PretrainConfig, index_path: str | Path, root: str | Path, workers: int
) -> tuple[DataLoader, DataLoader | None, list[str]]:
    train = WindowDataset(index_path, root, split="train")
    loader = DataLoader(
        train,
        batch_size=config.optim.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=workers,
        worker_init_fn=worker_init_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )

    try:
        validation = WindowDataset(index_path, root, split="val", classes=train.classes)
        probe = DataLoader(
            validation,
            batch_size=config.optim.batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
        )
    except ValueError:
        probe = None
    return loader, probe, train.classes


class Trainer:
    def __init__(
        self,
        config: PretrainConfig,
        index_path: str | Path,
        root: str | Path,
        device: str = "cuda",
        workers: int = 2,
        noise_bank: NoiseBank | None = None,
    ):
        self.config = config
        self.device = torch.device(device)

        self.loader, self.probe, self.classes = build_loaders(config, index_path, root, workers)
        self.frontend = LogMelFrontEnd(config.features, config.data.sample_rate).to(self.device)
        self.pipeline = Pipeline(config.augment, noise_bank)
        self.model = MoCo(config.moco).to(self.device)

        parameters = list(self.model.encoder_q.parameters())
        self.demon: DemonConsistency | None = None
        if config.demon_loss.enabled:
            frame = int(round(config.data.segment_seconds * config.data.sample_rate))
            self.demon = DemonConsistency(
                config.demon_loss, config.moco.dim, config.data.sample_rate, frame
            ).to(self.device)
            parameters += list(self.demon.parameters())

        optim = config.optim
        if optim.optimizer == "adamw":
            self.optimiser = torch.optim.AdamW(
                parameters, lr=optim.lr, weight_decay=optim.weight_decay
            )
        else:
            self.optimiser = torch.optim.SGD(
                parameters, lr=optim.lr, momentum=optim.momentum, weight_decay=optim.weight_decay
            )

        self.generator = torch.Generator(device="cpu").manual_seed(config.seed)
        self.start_epoch = 0
        self.history: list[dict[str, Any]] = []

    def views(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.pipeline.views(waveform, self.config.data.sample_rate, self.generator)

    def step(self, waveform: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        wave_q, wave_k = self.views(waveform)
        logits, labels, query, key = self.model(self.frontend(wave_q), self.frontend(wave_k))

        loss = info_nce(logits, labels)
        stats = {"info_nce": float(loss.detach())}

        if self.demon is not None:
            auxiliary, demon_stats = self.demon(query, key, wave_q, wave_k)
            loss = loss + auxiliary
            stats.update(demon_stats)

        self.model.enqueue(key)
        stats["loss"] = float(loss.detach())
        stats["accuracy"] = float((logits.argmax(dim=1) == labels).float().mean())
        return loss, stats

    @torch.no_grad()
    def embed_split(self, loader: DataLoader, limit: int = 2048):
        self.model.eval()
        embeddings, labels = [], []
        seen = 0
        for waveform, label in loader:
            waveform = waveform.to(self.device, non_blocking=True)
            embeddings.append(self.model.embed(self.frontend(waveform)).cpu())
            labels.append(label)
            seen += label.shape[0]
            if seen >= limit:
                break
        self.model.train()
        if not embeddings:
            return torch.empty(0), torch.empty(0, dtype=torch.long)
        return torch.cat(embeddings), torch.cat(labels)

    def monitor(self) -> dict[str, float]:
        if self.probe is None:
            return {}
        train_embeddings, train_labels = self.embed_split(self.loader)
        probe_embeddings, probe_labels = self.embed_split(self.probe)
        accuracy = knn_accuracy(train_embeddings, train_labels, probe_embeddings, probe_labels)
        return {"knn_accuracy": accuracy}

    def train(self, epochs: int | None = None, log: Any = None) -> list[dict[str, Any]]:
        optim = self.config.optim
        total = epochs or optim.epochs
        chance = 1.0 / max(1, len(self.classes))

        for epoch in range(self.start_epoch, total):
            learning_rate = cosine_lr(epoch, total, optim.warmup_epochs, optim.lr)
            for group in self.optimiser.param_groups:
                group["lr"] = learning_rate

            started = time.time()
            running: dict[str, float] = {}
            batches = 0
            for waveform, _ in self.loader:
                waveform = waveform.to(self.device, non_blocking=True)
                loss, stats = self.step(waveform)

                self.optimiser.zero_grad(set_to_none=True)
                loss.backward()
                self.optimiser.step()

                for key, value in stats.items():
                    running[key] = running.get(key, 0.0) + value
                batches += 1

            record = {k: v / max(1, batches) for k, v in running.items()}
            record.update({"epoch": epoch, "lr": learning_rate, "seconds": time.time() - started})

            with torch.no_grad():
                waveform, _ = next(iter(self.loader))
                wave_q, wave_k = self.views(waveform.to(self.device))
                query = self.model.encoder_q(self.frontend(wave_q))
                key = self.model.encoder_k(self.frontend(wave_k))
                record.update(health(query, key))

            record.update(self.monitor())
            record["collapsed"] = is_collapsed(
                record, record.get("knn_accuracy", float("nan")), chance
            )
            record["chance"] = chance

            self.history.append(record)
            if log is not None:
                log(record)
            self.save(epoch)

            if record["collapsed"]:
                print(
                    f"epoch {epoch}: collapse detected — rank fraction "
                    f"{record['rank_fraction']:.4f}, k-NN {record.get('knn_accuracy', float('nan')):.4f} "
                    f"against chance {chance:.4f}. Stopping rather than burning compute."
                )
                break
        return self.history

    def save(self, epoch: int) -> Path:
        directory = resolve(self.config.checkpoint_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "last.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model": self.model.state_dict(),
                "optimiser": self.optimiser.state_dict(),
                "demon": self.demon.state_dict() if self.demon else None,
                "history": self.history,
                "config": self.config.name,
                "classes": self.classes,
            },
            path,
        )
        (directory / "history.json").write_text(
            json.dumps(self.history, indent=2), encoding="utf-8"
        )
        return path

    def resume(self, path: str | Path) -> int:
        checkpoint = torch.load(resolve(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"])
        self.optimiser.load_state_dict(checkpoint["optimiser"])
        if self.demon is not None and checkpoint.get("demon"):
            self.demon.load_state_dict(checkpoint["demon"])
        self.history = checkpoint.get("history", [])
        self.start_epoch = checkpoint["epoch"]
        return self.start_epoch
