# PHASE

Physics-Augmented Self-supervised Encoding for underwater acoustic target recognition.

MoCo-v2 pretraining where the two contrastive views are the same source clip rendered
through two independently sampled ocean propagation channels — Lloyd's Mirror surface
interference, multipath, transmission loss, sea-state noise, and Doppler. The encoder is
driven to discard channel-dependent nuisance variation and keep ship-intrinsic structure.
A DEMON-consistency term protects propeller shaft- and blade-rate modulation from being
destroyed by augmentation.

`PLAN.md` is the roadmap. `CLAUDE.md` is the rulebook. `state.md` says where the work is.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # source .venv/bin/activate on Linux and macOS
pip install -e ".[dev,data]"
pytest
```

On Colab or Kaggle, `bash scripts/00_setup_env.sh` keeps the preinstalled CUDA torch and
installs the rest.

## Layout

| Path | What lives there |
|---|---|
| `configs/` | Every hyperparameter. Nothing is hardcoded in a training script. |
| `src/phase/` | The package: data, features, augment, models, ssl, eval, viz. |
| `scripts/` | Numbered entry points, run in order. |
| `tests/` | Leakage, manifest, augmentation, and config invariants. |
| `docs/` | `decisions.md` and `results_log.md`. |
| `results/` | Tables and figures. |

`data/`, `runs/`, and `checkpoints/` are gitignored.

## Non-negotiables

Splits are grouped by recording, never by window. Every headline number carries a
recording-ID shuffle control and three seeds. DS3500 has no recording identity and never
produces a label-efficiency number.
