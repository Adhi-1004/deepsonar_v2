# PHASE — state

Updated: 2026-11-10

## Current phase

Phases 0-5 complete and audited. **Phase 6 (MoCo-v2 pretraining) accelerated and active.**
Phase 7 (Evaluation suite) and Phase 8 (Visualization & Interpretability suite) implementations complete.

## Done

Phases 0-4 complete and audited. Decisions D-001 to D-022 in `docs/decisions.md`.

- **Phase 0:** Scaffold, dependencies, seeding, config system, test suite.
- **Phase 1:** Data acquisition and DeepShip audit (47.22h, 609 transits, 4 classes).
- **Phase 2:** Stratified group splits by vessel (`recording_id`), leakage test passing.
- **Phase 3:** DC removal & FLAC re-encoding, feature pipelines (log-Mel, CQT, DEMON, LOFAR), and ESC-50 dry run passed (58.83% accuracy, 57.04% macro-F1 on Colab T4).
- **Phase 4:** Physics augmentations implemented (Lloyd's Mirror, Multipath, Transmission Loss, Noise Mixing, Doppler) with DEMON modulation preservation verified.
- **Phase 5:** Physics validation against BELLHOP / DS3500 completed. Documented negative result in deep water (D-021: Lloyd's Mirror diverged below 100 Hz; bounded to shallow water, abstract updated).
- **Phase 6 Acceleration & Fixes (D-022):**
  - Integrated PyTorch Automatic Mixed Precision (AMP FP16) with GradScaler.
  - Eliminated redundant 960k-point front-end FFT highpass filter.
  - Throttled k-NN evaluation to every 5 epochs with 512 samples.
  - Live Google Drive checkpointing via directory symlink in Colab.
  - Runtime reduced from >2 hours per model down to ~10-12 minutes per model on T4.
- **Phase 7 Implementation:**
  - `src/phase/eval/linear_probe.py`: Linear probing on frozen MoCo representations.
  - `src/phase/eval/label_efficiency.py`: Evaluation across label fractions (1% to 100%, 3 seeds, vessel-grouped).
  - `src/phase/eval/leakage_control.py`: Recording-ID shuffle control (Rule 2).
  - `src/phase/eval/cross_dataset.py`: Cross-dataset evaluation between DeepShip and ShipsEar.
  - `src/phase/models/transfer.py`: Transfer baselines (ImageNet & scratch).
  - `scripts/06_evaluate.py`: End-to-end evaluation runner.
- **Phase 8 Implementation:**
  - `src/phase/viz/embeddings.py`: Dual-coloured t-SNE (by class vs by recording ID, Rule 3).
  - `src/phase/viz/gradcam.py`: Spectrogram Grad-CAM on ResNet layer 4.
  - `src/phase/viz/curves.py`: Label-efficiency curve plotting.
  - `scripts/07_make_figures.py`: Updated with flags for all visualizations.
  - `notebooks/colab_pretrain.ipynb`: Rewritten and optimized for fast execution.
  - `notebooks/colab_eval.ipynb`: Complete Colab evaluation pipeline.

## In Flight

- Pretraining `moco_physics` (and baseline `moco_generic`) in Colab using the updated notebook.
