# Results log

One entry per reported number. Every entry carries the config path, git SHA, seeds, and
the recording-ID shuffle control alongside the headline metric. Numbers without three
seeds and without a leakage control do not belong here.

| Date | Phase | Config | Git SHA | Seeds | Metric | Value (mean ± std) | Leakage control | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-08-28 | 3 | `configs/eval/finetune.yaml` | `e9ff950` | 0, 1, 2 | accuracy | **0.5883 ± 0.0076** | split grouped by source clip, 0 straddling | ESC-50 dry run, Colab T4 |
| 2026-08-28 | 3 | `configs/eval/finetune.yaml` | `e9ff950` | 0, 1, 2 | macro-F1 | **0.5704 ± 0.0096** | as above | as above |

---

## ESC-50 dry run — Phase 3 exit criterion, closed

**PASS.** 58.83% accuracy on 50 classes is **29.4x the 2% chance level**, above the 55%
threshold fixed in advance. `results/tables/finetune_3seeds.json`.

Per-seed: 0.5900, 0.5950, 0.5800. **Seed spread of ±0.76%** — a stable trainer, which
matters more than the mean here, because the same trainer carries the linear probe and
fine-tune protocols in Phase 7.

**This is a plumbing test, not a result.** It exists so that when DeepShip training
misbehaves, the pipeline is not a suspect. It says nothing about the PHASE method.

**On the number being at the low end.** The expected band was 60-80%, so 58.8% sits just
under it. That is consistent with the setup rather than a defect:

- ResNet-18 trained **from scratch**, no ImageNet initialisation
- **No augmentation at all** in this baseline
- **1,200 training clips**, because a validation split is carved out of the 1,600 the
  standard ESC-50 protocol gives to training
- Published from-scratch CNNs on ESC-50 without augmentation sit around 50-65%; the 80%+
  figures in the literature come from AudioSet pretraining or heavy augmentation

Nothing here needs fixing before Phase 6. If Phase 7's supervised DeepShip baseline lands
far below its published range, revisit the recipe then, with this as the reference point.
