#!/usr/bin/env python
"""Phase 7: Full evaluation pipeline — linear probe, label efficiency, cross-dataset.

Usage examples
--------------
# Linear probe on a single checkpoint
python scripts/09_linear_probe.py \\
    --checkpoint checkpoints/moco_physics_demon/last.pt \\
    --cache      data/cache/deepship_fold_0_windows.npz \\
    --root       data/clean/deepship \\
    --config     configs/eval/linear_probe.yaml

# Label-efficiency sweep
python scripts/09_linear_probe.py --label-efficiency \\
    --checkpoint checkpoints/moco_physics_demon/last.pt \\
    --cache      data/cache/deepship_fold_0_windows.npz \\
    --root       data/clean/deepship \\
    --config     configs/eval/label_efficiency.yaml

# Cross-dataset (train DeepShip → test ShipsEar)
python scripts/09_linear_probe.py --cross-dataset \\
    --checkpoint    checkpoints/moco_physics_demon/last.pt \\
    --cache         data/cache/deepship_fold_0_windows.npz \\
    --root          data/clean/deepship \\
    --target-cache  data/cache/shipsear_windows.npz \\
    --config        configs/eval/linear_probe.yaml

# Leakage control
python scripts/09_linear_probe.py --leakage-control \\
    --checkpoint checkpoints/moco_physics_demon/last.pt \\
    --cache      data/cache/deepship_fold_0_windows.npz \\
    --root       data/clean/deepship \\
    --config     configs/eval/linear_probe.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from phase.config import REPO_ROOT, EvalConfig, load_config
from phase.data import cache as data_cache
from phase.data.dataset import WindowDataset
from phase.data.manifest import resolve
from phase.eval import label_efficiency as le_module
from phase.eval import leakage_control as lc_module
from phase.eval import cross_dataset as cd_module
from phase.eval.linear_probe import train_linear_probe
from phase.eval.metrics import aggregate
from phase.features.frontend import LogMelFrontEnd
from phase.models.moco import MoCo
from phase.seed import set_seed
from phase.tracking import finish_run, init_run
from phase.config import WandbConfig, MocoConfig, FeatureConfig, load_yaml


# ---------------------------------------------------------------------------
# Embedding extraction from MoCo checkpoint
# ---------------------------------------------------------------------------


def extract_embeddings_from_checkpoint(
    checkpoint_path: str | Path,
    cache_path: str | Path,
    root: str | Path,
    device: str,
    batch_size: int = 256,
    workers: int = 2,
) -> tuple:
    """Load a MoCo checkpoint and embed all windows in the cache.

    Returns
    -------
    embeddings : torch.Tensor of shape (N, D)
    meta       : dict with labels, splits, recording_ids, classes
    """
    ckpt = torch.load(resolve(checkpoint_path), map_location=device, weights_only=False)

    # Reconstruct the MoCo config from the checkpoint name or the config file
    # We infer MocoConfig defaults; real usage passes the original config.
    # For robustness we just load the saved model directly.
    sample_config = MocoConfig()
    model = MoCo(sample_config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Feature config — read from paired config if present
    feature_config_path = resolve(checkpoint_path).parent / "feature_config.yaml"
    if feature_config_path.is_file():
        feature_config = load_config(str(feature_config_path), FeatureConfig)
    else:
        feature_config = load_config("configs/features/logmel.yaml", FeatureConfig)

    dataset = WindowDataset(cache_path, root)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=(device != "cpu"),
    )

    # Determine sample rate from dataset
    sample_rate = dataset.sample_rate
    frontend = LogMelFrontEnd(feature_config, sample_rate).to(device)

    all_emb = []
    with torch.no_grad():
        for waveforms, _ in loader:
            waveforms = waveforms.to(device)
            features = frontend(waveforms)
            emb = model.embed(features)
            all_emb.append(emb.cpu())

    embeddings = torch.cat(all_emb).numpy()
    meta = {
        "labels": [w["class"] for w in dataset.windows],
        "splits": [w["split"] for w in dataset.windows],
        "recording_ids": dataset.recording_ids(),
        "classes": dataset.classes,
    }
    return embeddings, meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 7 linear-probe evaluation")
    parser.add_argument("--config", default="configs/eval/linear_probe.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to MoCo last.pt")
    parser.add_argument("--cache", required=True, help="Window index (.npz or .json)")
    parser.add_argument("--root", required=True, help="Audio root directory")
    parser.add_argument("--target-cache", default=None, help="Target dataset cache (cross-dataset)")
    parser.add_argument("--target-root", default=None, help="Target dataset audio root")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--out", default="results/tables")
    parser.add_argument("--label-efficiency", action="store_true")
    parser.add_argument("--cross-dataset", action="store_true")
    parser.add_argument("--leakage-control", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config, EvalConfig)

    print(f"Extracting embeddings from {args.checkpoint} …")
    embeddings, meta = extract_embeddings_from_checkpoint(
        args.checkpoint, args.cache, args.root, args.device, workers=args.workers
    )
    print(f"  → embeddings {embeddings.shape}")

    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.checkpoint).parent.name  # e.g. moco_physics_demon

    # ---- Label efficiency sweep ----------------------------------------
    if args.label_efficiency:
        print("\nLabel-efficiency sweep …")
        result = le_module.run_label_efficiency(embeddings, meta, config, device=args.device)
        path = le_module.write(result, out_dir / f"{stem}_label_efficiency.json")
        print("\n" + le_module.render_table(result))
        print(f"\nWrote {path.relative_to(REPO_ROOT).as_posix()}")
        return 0

    # ---- Cross-dataset transfer -----------------------------------------
    if args.cross_dataset:
        if not args.target_cache:
            parser.error("--cross-dataset requires --target-cache")
        print("\nCross-dataset evaluation …")
        t_root = args.target_root or args.root
        target_embeddings, target_meta = extract_embeddings_from_checkpoint(
            args.checkpoint, args.target_cache, t_root, args.device, workers=args.workers
        )
        result = cd_module.run_cross_dataset(
            embeddings, meta, target_embeddings, target_meta,
            seeds=config.seeds, optim_config=config.optim, device=args.device,
        )
        path = cd_module.write(result, out_dir / f"{stem}_cross_dataset.json")
        print(f"\nTarget summary:")
        for k, v in result["target_summary"].items():
            print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")
        print(f"Wrote {path.relative_to(REPO_ROOT).as_posix()}")
        return 0

    # ---- Leakage control -----------------------------------------------
    if args.leakage_control:
        print("\nLeakage control experiment …")
        result = lc_module.run_leakage_control(
            embeddings, meta, config.optim,
            seeds=config.seeds, device=args.device,
        )
        path = lc_module.write(result, out_dir / f"{stem}_leakage_control.json")
        print("\nDelta (real − shuffled):")
        for metric, d in result["delta"].items():
            print(f"  {metric}: real={d['real_mean']:.4f}  shuffled={d['shuffled_mean']:.4f}  Δ={d['delta']:+.4f}")
        print(f"Wrote {path.relative_to(REPO_ROOT).as_posix()}")
        return 0

    # ---- Default: single linear probe ----------------------------------
    print(f"\nLinear probe (seeds={config.seeds}) …")
    runs = []
    for seed in config.seeds:
        set_seed(seed)
        run = init_run(
            config,
            WandbConfig(project=config.wandb.project, mode=args.wandb_mode,
                        tags=list(config.wandb.tags)),
            job_type="linear_probe",
            name=f"{stem}-seed{seed}",
        )
        result = train_linear_probe(
            embeddings, meta, config.optim, seed=seed, device=args.device,
            log=lambda r: run.log(r),
        )
        finish_run(run)
        scores = result.get("test", result["best_val"])
        print(f"  seed {seed}: " + " ".join(f"{k}={v:.4f}" for k, v in scores.items()
                                             if isinstance(v, float)))
        runs.append({k: v for k, v in scores.items() if isinstance(v, float)})

    summary = aggregate(runs)
    print()
    for key, stats in summary.items():
        print(f"  {key:20s} {stats['mean']:.4f} ± {stats['std']:.4f}")

    out_path = out_dir / f"{stem}_linear_probe_{len(config.seeds)}seeds.json"
    out_path.write_text(
        json.dumps({"seeds": config.seeds, "runs": runs, "summary": summary}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {out_path.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

