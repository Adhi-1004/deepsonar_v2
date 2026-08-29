# PHASE — state

Updated: 2026-08-28

## Current phase

Phase 5 — Physics validation against BELLHOP. **Complete. The result is negative.**

## Done

Phases 0-4 complete and audited. Decisions D-001 to D-021 in `docs/decisions.md`.

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

- **The Phase 5 result changes the paper's claims.** `PLAN.md` §0 states that a cheap
  analytic model "yields representations comparable to full ray-model simulation". That
  sentence is now falsified for deep water and has to be rewritten or dropped. The central
  label-efficiency claim is untouched.
- **ESC-50 dry run still has not converged** — one seed, 3 epochs, 32.25%. This is Phase 3's
  last exit criterion. Run in a terminal, not backgrounded, roughly 2.5-3 h:
  `.venv\Scripts\python.exe scripts/06_evaluate.py --cache data/cache/esc50_fold_0_logmel --seeds 0 1 2 --batch-size 16`
- **Augmentation cost on a real training GPU (D-020).** Measured 33 ms per view-window on
  the local MX450 against 142-166 ms on CPU. The 4.4x gain suggests memory-bandwidth bound,
  so a T4 should be comfortably faster, but that remains an extrapolation.
- **Nothing since commit `ba98799` is pushed.** The push failed: the stored credential is
  for `Rodhiq` while the repository belongs to `Adhi-1004`.

## Last result

`pytest` — 76 passed, 1 intentional skip. `ruff check` clean.

## Next

Phase 6 — MoCo-v2 pretraining, once the ESC-50 gate closes and the claim wording is fixed.
Three variants: `moco_generic`, `moco_physics`, `moco_physics_demon`. The generic arm is the
direct ablation of the whole idea and must be tuned as carefully as the physics arm.
