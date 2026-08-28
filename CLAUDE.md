# CLAUDE.md — PHASE Project Rules

Read this before every task. These are constraints, not suggestions.

## What this project is

PHASE: Physics-Augmented Self-supervised Encoding for underwater acoustic target
recognition. We pretrain an encoder with MoCo-v2 where the two contrastive views are
produced by passing the *same* source clip through *two independently sampled ocean
propagation channels* (Lloyd's Mirror surface interference, multipath, transmission loss,
sea-state noise, Doppler). The encoder learns propagation-invariant, ship-intrinsic
features. A DEMON-consistency term protects propeller shaft/blade modulation cues from
being destroyed by augmentation.

The claim we are testing: **physics-consistent positives beat generic augmentations for
label efficiency**, i.e. we match a supervised baseline using far fewer labels.

The claim we are NOT making: beating supervised SOTA at 100% labels. Published ShipsEar
accuracy already exceeds 99%. Do not tune toward that number.

## Non-negotiable correctness rules

1. **Split by recording, never by window.** Assign whole recordings to train/val/test
   BEFORE segmentation. Overlapping windows from one recording must never straddle a
   split. Any code path that shuffles segments before splitting is a bug.
2. **Run the recording-ID shuffle control** on every headline result. If a linear probe
   can decode recording identity from the embeddings nearly as well as it decodes ship
   class, the result is leakage, not learning. Report both numbers.
3. **t-SNE gets coloured twice** — once by class, once by recording. If the recording
   colouring is cleaner than the class colouring, stop and fix the pipeline.
4. **Three seeds minimum** on every reported number. Report mean ± std. Single-seed
   results do not go in the report.
5. **DS3500 has no recording IDs.** Filenames encode class and simulated position only.
   Never compute headline label-efficiency numbers on it. It is for physics validation
   and cross-dataset transfer only.
6. **No test-set peeking.** Validation set drives all model selection. Test is touched
   once per final configuration.

## Code style

- No comments in code. Idiomatic, human-looking, runs as written.
- Explanations go in commit messages and in `docs/`, not inline.
- Every experiment is driven by a YAML config in `configs/`. No hardcoded hyperparameters
  in training scripts.
- Seed torch, numpy, python random, and set cudnn deterministic in one `set_seed()` call.
- Log to Weights & Biases. Every run logs its full config, git SHA, and seed.

## Environment

Colab / Kaggle free tier is the target. T4 or P100, 16 GB. Design for that:

- ResNet-18 backbone, batch 128–256, MoCo queue decouples negatives from batch size.
- Checkpoint every epoch to Drive. Sessions get killed.
- Never train a Transformer from scratch. AST/ViT appear only as AudioSet-pretrained
  transfer baselines.

## Before you write code

- Check `state.md` for what's done and what's in flight.
- If a phase's exit criteria in `PLAN.md` are not met, do not start the next phase.
- If something in `PLAN.md` turns out to be wrong or infeasible, say so and propose an
  alternative. Do not silently work around it.

## Reporting discipline

When a result is bad, report it as bad. A negative result with a correct protocol is a
finding. A good result with a leaky split is worthless and will not survive review.
