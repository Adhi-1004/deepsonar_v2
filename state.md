# PHASE — state

Updated: 2026-08-28

## Current phase

Phase 4 — Physics augmentation library. **Complete, awaiting review.**

## Done

Phases 0-3 complete. Decisions D-001 to D-020 in `docs/decisions.md`.

**Phase 4.**

- `src/phase/augment/` — Lloyd's Mirror, multipath, transmission loss (Thorp absorption plus
  spreading), noise mixing (Wenz sea-state model plus recorded ambient), Doppler, and the
  generic baseline arm. All batched, device-agnostic torch.
- `base.py` defines `SpectralAugmentation`, whose subclasses return a per-item frequency
  response. `Pipeline` fuses consecutive spectral ops into **one** rFFT/irFFT pair.
- `compose.py` draws the two independent views that become MoCo's positive pair.
- `NoiseBank` enforces the D-011 contamination guard in code: it raises rather than serve a
  ShipsEar clip outside the held-out list.
- `notebooks/colab_eval.ipynb` — closes the outstanding Phase 3 criterion on a GPU, and
  benchmarks augmentation cost.

**The gate passes.** `tests/test_augment_preserves_demon.py` — 9 tests, no skips:

- the synthetic source carries the injected 6.2 Hz shaft and 24.8 Hz blade rates
- each physics transform preserves both within 1.5 Hz
- the full pipeline preserves both in **both** views
- **negative control**: a heavy pitch shift *fails* the same check, so the test has power
- forbidden augmentations cannot even be constructed

`results/figures/augmentations.png` shows Lloyd's Mirror producing textbook comb notches at
1.9, 3.7, 5.6 and 7.4 kHz, and the DEMON peak surviving every transform.

## Blocked

Nothing blocking Phase 5.

## Open — must close before Phase 6

- **Augmentation cost is unverified on GPU (D-020).** The full pipeline is ~100-170 ms per
  view-window on CPU, roughly 40 s per batch at 128. These are elementwise complex ops and a
  GPU should transform that, but local torch is CPU-only so it is an expectation, not a
  measurement. Section 8 of `colab_eval.ipynb` measures it.
- **ESC-50 dry run has not converged.** Carried from Phase 3; the same notebook closes it.
  Expected 60-80%; the run so far only reached 11.25% at 2 epochs on CPU.
- **Nothing is pushed to GitHub.** The repo holds only the initial commit, so the Colab
  notebook would clone an empty project.

## Last result

`pytest` — **76 passed, 1 skipped** (only the intentional ShipsEar `split_safe: false`
case). `ruff check` clean.

## Next

Phase 5 — validate the analytic augmenter against BELLHOP. D-006 established 2,223 exact
matched pairs across all 36 geometries, so this is a per-clip comparison at known range and
depth: spectral distance, DEMON peak preservation, and embedding cosine similarity, reported
by range and frequency band.
