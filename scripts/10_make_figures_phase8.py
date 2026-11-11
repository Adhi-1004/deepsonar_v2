#!/usr/bin/env python
"""Phase 8: Generate all interpretability and results figures.

Usage
-----
# All figures at once
python scripts/10_make_figures_phase8.py \\
    --checkpoint checkpoints/moco_physics_demon/last.pt \\
    --cache      data/cache/deepship_fold_0_windows.npz \\
    --root       data/clean/deepship \\
    --results    results/tables \\
    --out        results/figures

# Just embeddings (t-SNE / UMAP)
python scripts/10_make_figures_phase8.py --embeddings-only \\
    --checkpoint checkpoints/moco_physics_demon/last.pt \\
    --cache      data/cache/deepship_fold_0_windows.npz \\
    --root       data/clean/deepship \\
    --out        results/figures

# Just Grad-CAM
python scripts/10_make_figures_phase8.py --gradcam-only \\
    --checkpoint checkpoints/moco_physics_demon/last.pt \\
    --cache      data/cache/deepship_fold_0_windows.npz \\
    --root       data/clean/deepship \\
    --out        results/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from phase.config import REPO_ROOT, MocoConfig, FeatureConfig, load_config
from phase.data.dataset import WindowDataset
from phase.data.manifest import resolve
from phase.features.frontend import LogMelFrontEnd
from phase.models.moco import MoCo
from phase.models.resnet import SpectrogramClassifier
from phase.viz import embeddings as emb_module
from phase.viz import gradcam as gc_module
from phase.viz import curves as curve_module


# ---------------------------------------------------------------------------
# Shared embedding extraction
# ---------------------------------------------------------------------------


def _load_encoder(checkpoint_path: str | Path, device: str):
    ckpt = torch.load(resolve(checkpoint_path), map_location=device, weights_only=False)
    model = MoCo(MocoConfig()).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt.get("classes", [])


def _load_frontend(checkpoint_path: str | Path, sample_rate: int) -> LogMelFrontEnd:
    feature_config_path = resolve(checkpoint_path).parent / "feature_config.yaml"
    if feature_config_path.is_file():
        config = load_config(str(feature_config_path), FeatureConfig)
    else:
        config = load_config("configs/features/logmel.yaml", FeatureConfig)
    return LogMelFrontEnd(config, sample_rate)


# ---------------------------------------------------------------------------
# Embedding scatter plots (t-SNE / UMAP)
# ---------------------------------------------------------------------------


def make_embedding_figures(args) -> list[Path]:
    print("Extracting embeddings …")
    dataset = WindowDataset(args.cache, args.root)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False, num_workers=args.workers,
    )
    device = args.device
    encoder, _ = _load_encoder(args.checkpoint, device)
    frontend = _load_frontend(args.checkpoint, dataset.sample_rate).to(device)

    embeddings_arr, class_labels, rec_ids = emb_module.extract_embeddings(
        encoder, frontend, loader, device=device, limit=args.embed_limit
    )
    print(f"  → {embeddings_arr.shape[0]} embeddings, {len(set(class_labels))} classes, "
          f"{len(set(rec_ids))} recordings")

    stem = Path(args.checkpoint).parent.name
    made = []
    for method in ("tsne", "umap"):
        out = resolve(args.out) / f"{stem}_{method}.png"
        made.append(emb_module.embedding_panel(
            embeddings_arr, class_labels, rec_ids, out, method=method
        ))
        print(f"  wrote {out.relative_to(REPO_ROOT).as_posix()}")

    return made


# ---------------------------------------------------------------------------
# Grad-CAM panels
# ---------------------------------------------------------------------------


def make_gradcam_figures(args) -> list[Path]:
    print("Preparing Grad-CAM …")
    dataset = WindowDataset(args.cache, args.root)
    device = args.device

    # We need a classifier (not just the SSL encoder) for Grad-CAM.
    # Try to load a fine-tuned checkpoint; fall back to building a fresh head.
    finetune_path = resolve(args.checkpoint).parent / "finetuned.pt"
    if finetune_path.is_file():
        ckpt = torch.load(str(finetune_path), map_location=device, weights_only=False)
        n_classes = len(dataset.classes)
        model = SpectrogramClassifier("resnet18", n_classes).to(device)
        model.load_state_dict(ckpt["model"])
    else:
        print("  (no finetuned.pt found — using random linear head over frozen encoder)")
        encoder, _ = _load_encoder(args.checkpoint, device)
        n_classes = len(dataset.classes)
        model = SpectrogramClassifier("resnet18", n_classes).to(device)
        # Copy backbone weights from MoCo encoder
        with torch.no_grad():
            model.backbone.load_state_dict(encoder.encoder_q[0].state_dict(), strict=False)

    model.eval()
    frontend = _load_frontend(args.checkpoint, dataset.sample_rate).to(device)

    # Collect a few examples per class
    per_class: dict[str, list] = {c: [] for c in dataset.classes}
    for idx in range(min(500, len(dataset))):
        waveform, label_idx = dataset[idx]
        cls = dataset.classes[label_idx]
        if len(per_class[cls]) < 2:
            with torch.no_grad():
                wav_t = torch.from_numpy(waveform).unsqueeze(0).to(device)
                spec = frontend(wav_t).squeeze(0).squeeze(0).cpu().numpy()
            per_class[cls].append((spec, cls))
        if all(len(v) >= 2 for v in per_class.values()):
            break

    specs, true_classes = [], []
    for cls in dataset.classes:
        for spec, label in per_class[cls]:
            specs.append(spec)
            true_classes.append(label)

    stem = Path(args.checkpoint).parent.name
    out = resolve(args.out) / f"{stem}_gradcam.png"
    made = gc_module.gradcam_panel(
        model, specs, dataset.classes, true_classes, dataset.sample_rate, out
    )
    print(f"  wrote {out.relative_to(REPO_ROOT).as_posix()}")
    return [made]


# ---------------------------------------------------------------------------
# Label-efficiency and monitor curves from saved JSON results
# ---------------------------------------------------------------------------


def make_results_figures(args) -> list[Path]:
    results_dir = resolve(args.results)
    out_dir = resolve(args.out)
    made = []

    # --- Label-efficiency curve (gathers all *_label_efficiency.json) ---
    le_files = sorted(results_dir.glob("*_label_efficiency.json"))
    if le_files:
        all_results = {}
        for path in le_files:
            data = json.loads(path.read_text())
            method = path.stem.replace("_label_efficiency", "")
            all_results[method] = data

        for metric in ("accuracy", "macro_f1"):
            out = out_dir / f"label_efficiency_{metric}.png"
            made.append(curve_module.label_efficiency_curve(all_results, out, metric=metric))
            print(f"  wrote {out.relative_to(REPO_ROOT).as_posix()}")

    # --- Training monitor curves (gathers checkpoints/*/history.json) ---
    for history_file in sorted(resolve("checkpoints").glob("*/history.json")):
        history = json.loads(history_file.read_text())
        name = history_file.parent.name
        out = out_dir / f"{name}_monitors.png"
        made.append(curve_module.training_monitor_curves(history, out, name=name))
        print(f"  wrote {out.relative_to(REPO_ROOT).as_posix()}")

    # --- Cross-dataset bar chart ---
    cd_files = sorted(results_dir.glob("*_cross_dataset.json"))
    if cd_files:
        cd_results = {}
        for path in cd_files:
            data = json.loads(path.read_text())
            method = path.stem.replace("_cross_dataset", "")
            cd_results[method] = data
        for metric in ("accuracy", "macro_f1"):
            out = out_dir / f"cross_dataset_{metric}.png"
            made.append(curve_module.cross_dataset_bar(cd_results, out, metric=metric))
            print(f"  wrote {out.relative_to(REPO_ROOT).as_posix()}")

    return made


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 8 interpretability figures")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--results", default="results/tables")
    parser.add_argument("--out", default="results/figures")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--embed-limit", type=int, default=4096)
    parser.add_argument("--embeddings-only", action="store_true")
    parser.add_argument("--gradcam-only", action="store_true")
    parser.add_argument("--results-only", action="store_true")
    args = parser.parse_args(argv)

    resolve(args.out).mkdir(parents=True, exist_ok=True)
    made: list[Path] = []

    if args.embeddings_only:
        made += make_embedding_figures(args)
    elif args.gradcam_only:
        made += make_gradcam_figures(args)
    elif args.results_only:
        made += make_results_figures(args)
    else:
        # All figures
        made += make_embedding_figures(args)
        made += make_gradcam_figures(args)
        made += make_results_figures(args)

    print(f"\nTotal figures written: {len(made)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
