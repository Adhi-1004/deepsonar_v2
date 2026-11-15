from __future__ import annotations

import argparse
import json

from phase.config import REPO_ROOT, FeatureConfig, load_config
from phase.data.manifest import resolve
from phase.viz import spectrograms


def render_augmentations(args) -> int:
    import time

    import soundfile as sf
    import torch

    from phase.augment import Pipeline, build_op
    from phase.config import AugmentConfig
    from phase.viz import spectrograms

    config = load_config("configs/augment/physics.yaml", AugmentConfig)
    feature = load_config("configs/features/demon.yaml", FeatureConfig)

    report = json.loads(resolve(args.report).read_text(encoding="utf-8"))
    row = sorted(report["rows"], key=lambda r: r["ac_rms"])[len(report["rows"]) // 2]
    path = (resolve(args.clean) / row["path"]).with_suffix(".flac")

    seconds = 20.0
    audio, rate = sf.read(str(path), frames=int(seconds * 32000), dtype="float32")

    ops = {
        op.name: build_op(op) for op in config.ops if op.name not in {"random_crop", "time_mask"}
    }
    made = spectrograms.augmentation_panel(
        audio, rate, ops, feature.params, f"{args.out}/augmentations.png"
    )
    print(f"wrote {made.relative_to(REPO_ROOT).as_posix()}")

    pipeline = Pipeline(config)
    generator = torch.Generator().manual_seed(0)
    batch = torch.from_numpy(audio).unsqueeze(0).repeat(8, 1)
    pipeline.views(batch, rate, generator)
    start = time.time()
    for _ in range(3):
        first, second = pipeline.views(batch, rate, generator)
    elapsed = (time.time() - start) / 3

    print("sampled geometry for view 1, first item:")
    lloyd = ops["lloyds_mirror"]
    drawn = lloyd.sample_params(3, rate, generator, batch.device)
    for index in range(3):
        print(
            f"  view {index}: source {drawn['source_depth'][index]:.1f} m, "
            f"receiver {drawn['receiver_depth'][index]:.1f} m, "
            f"range {drawn['range'][index]:.0f} m, "
            f"delay {lloyd.delay(drawn)[index] * 1000:.3f} ms"
        )
    print()
    print(
        f"pipeline cost (CPU): {elapsed:.2f} s per batch of 8 x 2 views "
        f"= {1000 * elapsed / 16:.1f} ms per view-window at {seconds:.0f} s"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render Phase 3 sanity figures")
    parser.add_argument("--raw", default="data/raw/deepship/deepship-raw")
    parser.add_argument("--clean", default="data/clean/deepship")
    parser.add_argument("--report", default="data/manifests/deepship_clean_report.json")
    parser.add_argument("--out", default="results/figures")
    parser.add_argument("--augmentations", action="store_true")
    parser.add_argument("--label-efficiency", action="store_true")
    parser.add_argument("--embeddings", action="store_true")
    parser.add_argument("--gradcam", action="store_true")
    parser.add_argument("--cache", default="data/cache/deepship_fold_0_logmel")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args(argv)

    if args.augmentations:
        return render_augmentations(args)

    if args.label_efficiency:
        from phase.viz.curves import label_efficiency_curve

        results_dir = resolve("results/tables")
        results = {}
        for file in sorted(results_dir.glob("label_efficiency_*.json")):
            name = file.stem.replace("label_efficiency_", "").split("_")[0]
            results[name] = json.loads(file.read_text(encoding="utf-8"))
        if results:
            out_img = label_efficiency_curve(results, f"{args.out}/label_efficiency.png")
            print(f"wrote {out_img.relative_to(REPO_ROOT).as_posix()}")
        else:
            print("no label efficiency result files found in results/tables/")
        return 0

    if args.embeddings:
        from phase.data import cache
        from phase.eval.linear_probe import extract_embeddings
        from phase.models.transfer import load_classifier
        from phase.viz.embeddings import plot_embeddings_dual

        features, meta = cache.load(args.cache)
        model = load_classifier("resnet18", len(set(meta["labels"])), checkpoint_path=args.checkpoint)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(torch.from_numpy(features[:1000]).float(), torch.zeros(1000)), batch_size=128)
        embs, _ = extract_embeddings(model.backbone, loader, device)
        out_img = plot_embeddings_dual(
            embs.numpy(),
            meta["labels"][:1000],
            meta["recording_ids"][:1000],
            f"{args.out}/embeddings_tsne.png",
        )
        print(f"wrote {out_img.relative_to(REPO_ROOT).as_posix()}")
        return 0

    if args.gradcam:
        from phase.data import cache
        from phase.models.transfer import load_classifier
        from phase.viz.gradcam import GradCAM, plot_gradcam

        features, meta = cache.load(args.cache)
        model = load_classifier("resnet18", len(set(meta["labels"])), checkpoint_path=args.checkpoint)
        cam = GradCAM(model, model.backbone.layer4)
        sample = torch.from_numpy(features[0]).float()
        heatmap = cam.generate(sample)
        out_img = plot_gradcam(features[0], heatmap, f"{args.out}/gradcam.png")
        print(f"wrote {out_img.relative_to(REPO_ROOT).as_posix()}")
        return 0

    report = json.loads(resolve(args.report).read_text(encoding="utf-8"))
    worst = max(report["rows"], key=lambda r: r["low_freq_fraction"])
    typical = sorted(report["rows"], key=lambda r: r["ac_rms"])[len(report["rows"]) // 2]

    made = []
    for label, row in [("worst_drift", worst), ("typical", typical)]:
        made.append(
            spectrograms.dc_before_after(
                resolve(args.raw) / row["path"],
                (resolve(args.clean) / row["path"]).with_suffix(".flac"),
                f"{args.out}/dc_removal_{label}.png",
            )
        )

    configs = {
        name: load_config(f"configs/features/{name}.yaml", FeatureConfig)
        for name in ("logmel", "cqt", "demon")
    }
    made.append(
        spectrograms.feature_panel(
            (resolve(args.clean) / typical["path"]).with_suffix(".flac"),
            configs,
            f"{args.out}/features_deepship.png",
        )
    )

    for path in made:
        print(f"wrote {path.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
