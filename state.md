# PHASE — state

Updated: 2026-11-11

## Current phase

Phases 0–5 complete. Phase 6 scaffold complete (Trainer + all SSL code
in place). **Phases 7–8 code complete — ready to run once Phase 6
checkpoints exist.**

## Done

Phases 0–5 complete and audited. Decisions D-001 to D-021 in `docs/decisions.md`.

**Phase 6 (code).** `src/phase/ssl/train_moco.py` Trainer, `src/phase/ssl/losses.py`,
`src/phase/ssl/demon_consistency.py`, `src/phase/ssl/monitors.py`, `scripts/05_pretrain.py`
all implemented and passing tests. Awaiting actual training run on GPU.

**Phase 7 (code).** All evaluation modules written and passing the full test suite
(76 passed, 1 skip):
- `src/phase/eval/linear_probe.py` — frozen-encoder linear head, label-fraction sub-sampling
  grouped by recording_id, cosine LR, val-best checkpoint.
- `src/phase/eval/label_efficiency.py` — sweep over `label_fractions` × `seeds`, aggregate
  summary with mean ± std per fraction, `render_table()` for console output.
- `src/phase/eval/cross_dataset.py` — train on source splits, evaluate on entire target
  dataset; multi-seed sweep; writes `target_summary` and `source_summary`.
- `src/phase/eval/leakage_control.py` — within-class permutation of `recording_ids`;
  compares real grouped split vs shuffled; Δ table shows how much grouping helps.
- `src/phase/data/__init__.py` — `load()`/`save()` helpers for `.npz` feature cache.
- `scripts/09_linear_probe.py` — CLI wrapping all four protocols; `--label-efficiency`,
  `--cross-dataset`, `--leakage-control` flags.

**Phase 8 (code).** Interpretability and figure generation:
- `src/phase/viz/embeddings.py` — `extract_embeddings()`, t-SNE / UMAP dual scatter
  (class + recording), `per_class_uniformity()`.
- `src/phase/viz/gradcam.py` — `GradCAM` hook-based class, `gradcam_panel()` grid figure.
- `src/phase/viz/curves.py` — `label_efficiency_curve()`, `training_monitor_curves()`,
  `confusion_matrix_figure()`, `cross_dataset_bar()`; consistent colour palette.
- `src/phase/models/transfer.py` — `TransferEncoder` for ImageNet ResNet and AudioSet PANN,
  with graceful fallback if timm weights unavailable.
- `scripts/10_make_figures_phase8.py` — one command for all Phase 8 figures.
- `configs/eval/cross_dataset.yaml` added.

**Phase 5.** `src/phase/eval/physics_validation.py` and `scripts/08_validate_physics.py`.
216 matched pairs, six per geometry across all 36, each rendered at its **actual** labelled
range and receiver depth rather than a sampled one.

**The analytic channel model does not approximate BELLHOP. It makes agreement worse.**

| Variant | LSD to BELLHOP, dB | 10-100 Hz |
|---|---|---|
| none (baseline) | **6.348** | **6.874** |
| lloyd | 9.786 | 17.483 |
| tl | 6.525 | 6.764 |
| lloyd+tl | 9.823 | 15.782 |
| lloyd+tl+multipath | 10.420 | 15.533 |

Lloyd's Mirror costs about +3.4 dB overall and +10.6 dB below 100 Hz, flat across every
receiver depth. Transmission loss is roughly neutral. See D-021 for why this is a modelling
mismatch rather than a bug: Lloyd's Mirror is a shallow-water surface-interference model and
DS3500 renders 3500 m deep water with a real sound-speed profile and shadow zones.

**The two geometries do not overlap.** The augmenter samples receiver depths of 10-60 m;
DS3500's shallowest is 100 m. This test measures extrapolation, so the shallow regime the
model was written for is left **unvalidated, not validated**.

Figure: `results/figures/physics_validation.png`.

## Blocked

Nothing blocked, but two things must be settled before Phase 6.

## Open — must close before Phase 6

- ~~The Phase 5 result changes the paper's claims~~ **DONE.** The abstract in `PLAN.md` §0
  now reports the negative result and bounds the analytic model to shallow water. The risk
  table records that this risk occurred. The central label-efficiency claim is untouched.
- ~~ESC-50 dry run~~ **CLOSED 2026-08-28.** 58.83% ± 0.76% accuracy, 57.04% ± 0.96%
  macro-F1, three seeds on a Colab T4 — 29.4x the 2% chance level, above the 55% threshold
  fixed in advance. Phase 3 is complete. See `docs/results_log.md`.
- **Augmentation cost on a real training GPU (D-020).** Measured 33 ms per view-window on
  the local MX450 against 142-166 ms on CPU. The 4.4x gain suggests memory-bandwidth bound,
  so a T4 should be comfortably faster, but that remains an extrapolation.
- ~~**Nothing since commit `ba98799` is pushed.** The push failed: the stored credential is
  for `Rodhiq` while the repository belongs to `Adhi-1004`.~~ **DONE.**

## Last result

`pytest` — 76 passed, 1 intentional skip. `ruff check` clean.

## Next

Phase 6 — MoCo-v2 pretraining. Both blockers are cleared: the ESC-50 gate passed and
the abstract wording is corrected.
Three variants: `moco_generic`, `moco_physics`, `moco_physics_demon`. The generic arm is the
direct ablation of the whole idea and must be tuned as carefully as the physics arm.
