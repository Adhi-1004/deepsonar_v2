# PHASE — Full Project Plan

Physics-Augmented Self-supervised Encoding for Underwater Acoustic Target Recognition

Read `CLAUDE.md` first. This document is the phased roadmap. Work top to bottom. Each
phase has exit criteria; do not advance until they are met.

---

## 0. Abstract (working version)

Underwater acoustic target recognition from passive sonar is constrained by the scarcity
of labelled ship-radiated-noise recordings, while unlabelled hydrophone data are
abundant. We present PHASE, a physics-augmented self-supervised framework that reduces
labelled-data requirements for vessel classification by learning propagation-invariant
representations. The central idea is to generate contrastive positive pairs not through
generic image-style augmentations but through physically grounded ocean-channel
transformations: the same source clip is rendered through independently sampled
propagation conditions — Lloyd's Mirror surface interference, multipath delay structure,
transmission-loss scaling, Doppler shift, and sea-state noise mixing — so that a
momentum-contrastive encoder is driven to discard channel-dependent nuisance variation
while preserving source signatures such as propeller shaft- and blade-rate modulation. A
DEMON-consistency regulariser explicitly protects these envelope-modulation cues from
being destroyed by augmentation. We further validate the augmentation itself against
DS3500, a BELLHOP ray-theory rendering of ShipsEar across thirty-six simulated
source-receiver geometries, showing that a computationally cheap analytic channel model
yields representations comparable to full ray-model simulation. Pretraining on unlabelled
ship recordings and evaluating on DeepShip and ShipsEar under recording-level splits that
eliminate the recording-identity leakage recently shown to inflate results in this
domain, we report label-efficiency curves from one to one hundred percent of labels
against supervised, ImageNet-transfer, AudioSet-transfer, and generic-augmentation
contrastive baselines. Ablations isolate the contribution of each physics augmentation and
of the DEMON regulariser. Results indicate that encoding domain physics into the
self-supervised objective is an effective and compute-modest route to data-efficient
passive sonar classification.

**Short version (100 words).** Passive-sonar ship classification is bottlenecked by scarce
labels but rich unlabelled hydrophone data. PHASE is a physics-augmented contrastive
framework that defines self-supervised positive pairs by rendering the same clip through
independently sampled ocean channels — Lloyd's Mirror interference, multipath,
transmission loss, and noise mixing — so a MoCo encoder learns propagation-invariant,
ship-intrinsic features, with a DEMON-consistency term protecting propeller-modulation
cues. Validated against BELLHOP ray-model renderings and evaluated on DeepShip and
ShipsEar with leakage-free recording-level splits, PHASE improves label efficiency and
reduces cross-dataset degradation, showing that embedding ocean physics into the
contrastive objective yields data-efficient underwater acoustic target recognition.

---

## 1. Repository layout

Create exactly this. Do not invent extra top-level directories.

```
phase/
  CLAUDE.md
  PLAN.md
  state.md
  README.md
  pyproject.toml
  .gitignore
  configs/
    data/
      deepship.yaml
      shipsear_ds3500.yaml
      qiandaoear.yaml
      esc50.yaml
    features/
      logmel.yaml
      cqt.yaml
      demon.yaml
    augment/
      generic.yaml
      physics.yaml
      physics_ablation/
    pretrain/
      moco_generic.yaml
      moco_physics.yaml
      moco_physics_demon.yaml
    eval/
      linear_probe.yaml
      finetune.yaml
      label_efficiency.yaml
  src/phase/
    __init__.py
    seed.py
    config.py
    data/
      download.py
      verify.py
      manifest.py
      splits.py
      segment.py
      dataset.py
    features/
      logmel.py
      cqt.py
      demon.py
      lofar.py
    augment/
      base.py
      generic.py
      lloyds_mirror.py
      multipath.py
      transmission_loss.py
      noise_mixing.py
      doppler.py
      compose.py
    models/
      resnet.py
      projection.py
      moco.py
      transfer.py
    ssl/
      train_moco.py
      losses.py
      demon_consistency.py
      monitors.py
    eval/
      linear_probe.py
      finetune.py
      label_efficiency.py
      cross_dataset.py
      leakage_control.py
      metrics.py
    viz/
      embeddings.py
      gradcam.py
      spectrograms.py
      curves.py
  scripts/
    00_setup_env.sh
    01_fetch_data.sh
    02_build_manifests.py
    03_build_splits.py
    04_segment_and_cache.py
    05_pretrain.py
    06_evaluate.py
    07_make_figures.py
  tests/
    test_splits_no_leakage.py
    test_augment_preserves_demon.py
    test_manifest_integrity.py
    test_config_roundtrip.py
  notebooks/
    colab_pretrain.ipynb
    colab_eval.ipynb
  data/            # gitignored
    raw/
    manifests/
    splits/
    cache/
  runs/            # gitignored
  checkpoints/     # gitignored
  results/
    tables/
    figures/
  docs/
    decisions.md
    results_log.md
```

`state.md` is a running log: current phase, what is done, what is blocked, what the last
result was. Update it at the end of every working session.

---

## 2. Data sources

| Dataset | Role | Access | Notes |
|---|---|---|---|
| DeepShip | Primary train/eval | Kaggle `tangqiji/deepship-raw` | Gate passed 2026-08-28: 609 transits, 47.22 h, 4/4 spec checks. See D-007. Cite Irfan et al. 2021, ESWA 183:115270 |
| ShipsEar + DS3500 | Physics validation, cross-dataset | HF `peng7554/DS3500`, CC-BY-4.0 | 1948 5-s segments @16 kHz, classes A–E (345/235/785/395/188). DS3500 = same signals via BELLHOP, 36 positions, range 1–11 km step 2 km, depth 100–1100 m step 200 m |
| ~~QiandaoEar22~~ | Dropped | IEEE DataPort paywall | Unobtainable. Ambient noise re-sourced to ShipsEar class E + Wenz synthetic. See D-011 |
| ESC-50 | Pipeline scaffolding, transfer baseline | Kaggle `mmoreaux/environmental-sound-classification-50` | Unblocked, use from day one |
| ONC | Optional unlabelled scale-up | Free account + API token at data.oceannetworks.ca | Only if Phase 6 shows pretraining data is the bottleneck |

Not used: SanctSound and NOAA raw buckets. Public and real, but no ship-type labels and
petabyte scale. Wrong cost/benefit for this timeline.

---

## 3. Phases

### Phase 0 — Scaffold

Set up the repo, dependencies, config system, seeding, W&B, and CI-style test stubs.

- `pyproject.toml` with pinned versions: torch, torchaudio, librosa, numpy, scipy,
  scikit-learn, timm, wandb, pyyaml, soundfile, nnAudio, umap-learn, matplotlib.
- `src/phase/seed.py` exposing `set_seed(seed)` covering torch, numpy, random, cudnn.
- `src/phase/config.py` loading YAML into typed dataclasses with validation.
- `scripts/00_setup_env.sh` reproducing the environment on Colab and Kaggle.
- Write the four tests in `tests/` as failing stubs now; they get implemented in the
  phases that make them meaningful.

**Exit:** `python -c "import phase"` works, a dummy config loads, W&B logs a test run.

---

### Phase 1 — Data acquisition and the DeepShip gate

This phase decides the shape of the whole project.

Implement `src/phase/data/verify.py` and run it on the DeepShip download. It must report:
file count, total hours, sample rates, directory structure depth, per-class file counts,
per-file durations, and whether filenames or folders encode individual ship transits.

```bash
kaggle datasets download -d vasundharauppuluri/deepship-main -p data/raw --unzip
python -m phase.data.verify --root data/raw/deepship --report data/manifests/deepship_audit.json
```

**Decision gate.** Compare against the published DeepShip spec: 47 h 4 min, 265 ships,
4 classes (cargo, passenger, tanker, tug), 32 kHz.

- **Full or near-full, per-transit files present** → DeepShip is primary. Continue as
  planned.
- **Subset but per-transit boundaries survive** → DeepShip stays primary, note the
  reduced scale in `docs/decisions.md`, and pull QiandaoEar22 in as additional unlabelled
  pretraining data.
- **Flat structure, no recoverable ship identity** → DeepShip cannot support recording-level
  splits. Demote it to evaluation-only, promote QiandaoEar22 to primary for pretraining,
  and build the labelled evaluation set from ONC via the API.

Record the decision and its evidence in `docs/decisions.md` before proceeding.

Also in this phase: download DS3500 from HuggingFace, QiandaoEar22 from the OneDrive link,
and ESC-50. Write `download.py` so all four are reproducible from one command.

**Exit:** all four datasets on disk, audit JSON committed, decision recorded.

---

### Phase 2 — Manifests and leakage-safe splits

Build a manifest before touching audio. One row per source file:
`path, dataset, class, recording_id, duration, sample_rate, channels`.

`recording_id` is the load-bearing field. For DeepShip it is the **ship**, recovered by
joining the transit directory to the class metafiles — not the transit, because one vessel
appears in up to 35 transits and two transits of one hull share a signature. See D-013.
For DS3500 it does not exist — set it null and mark the dataset `split_safe: false`.

Then `splits.py`: stratified by class, grouped by `recording_id`, using
`sklearn.model_selection.StratifiedGroupKFold`. Splits are written to
`data/splits/*.json` and are immutable once written. Never regenerate them mid-project.

Implement `tests/test_splits_no_leakage.py`: assert the intersection of `recording_id`
sets across train/val/test is empty, for every split file, every fold.

**Exit:** split files exist, the leakage test passes, class distribution per split is
logged.

---

### Phase 3 — Segmentation, features, and the ESC-50 dry run

Segment **after** splitting. DeepShip: 30 s windows, 15 s hop, matching the field-standard
protocol. ESC-50 and DS3500 come pre-segmented at 5 s.

Features, all implemented behind a common interface:

| Feature | Parameters |
|---|---|
| log-Mel (default) | n_fft 2048, hop 512, n_mels 128, fmin 10 Hz, fmax 8 kHz |
| CQT | 30 bins/octave, fmin at passband low |
| DEMON | envelope detection then FFT, 1–30 kHz analysis band |
| LOFAR | 50 ms frame, 25 ms shift, Hanning, TPSW normalisation |

Per-sample z-score normalisation. Cache features as `.npy` under `data/cache/`.

Then run the entire pipeline end to end on ESC-50: segment, cache, train a small
supervised classifier, log to W&B. This is a plumbing test, not a result. It exists so
that when DeepShip training misbehaves you know the pipeline is not the cause.

**Exit:** ESC-50 supervised baseline trains and reaches sane accuracy; feature cache
reproducible from config; spectrogram sanity plots in `results/figures/`.

---

### Phase 4 — Physics augmentation library

The scientific core. Each augmentation is a callable taking a waveform and a sampled
parameter dict, returning a waveform.

**Identity-preserving, use as positives:**

- **Lloyd's Mirror** — comb filter from direct plus surface-reflected path.
  `|H(f)|² = 2[1 + cos(2π f Δτ)]` where `Δτ` follows from the path-length difference given
  source depth `d`, receiver depth `z`, range `r`. Sample geometry independently per view.
- **Multipath** — delay-and-sum with a small number of sampled arrivals and decaying
  amplitudes.
- **Transmission loss** — frequency-dependent rescaling.
- **Ambient noise mixing** — inject ShipsEar class-E ambient or Wenz-curve sea-state noise
  at controlled SNR. Class-E clips come only from recordings held out of every ShipsEar
  split.
- **Doppler** — resample within a few percent.
- **Mild SpecAugment** — small time masking only.

**Identity-destroying, forbidden:** aggressive pitch shift, heavy frequency masking of the
tonal band, strong time warp. These move shaft and blade lines and destroy the label.

Implement `tests/test_augment_preserves_demon.py`: for each augmentation, compute the DEMON
spectrum before and after, assert the dominant modulation peaks persist within tolerance.
An augmentation that fails this test is not a valid positive.

**Exit:** all augmentations implemented and tested; a figure showing example spectrograms
under each transform with DEMON peaks annotated.

---

### Phase 5 — Physics validation against DS3500

This is the differentiator. Do it before pretraining, because it either strengthens or
kills the central premise.

DS3500 gives you the same 1,948 ShipsEar signals rendered through BELLHOP at 36
source-receiver geometries. Your Lloyd's Mirror augmenter is a cheap analytic
approximation of the same physics.

1. For matched geometries, apply your augmenter to the original ShipsEar clip with the
   corresponding range and depth parameters.
2. Compare against the BELLHOP rendering: spectral distance, DEMON peak preservation, and
   cosine similarity of embeddings from an ImageNet-pretrained encoder.
3. Report where the approximation holds and where it breaks — likely by range and by
   frequency band.

If agreement is reasonable, you have a validated augmenter and a strong figure. If it is
poor, that is still a result: it tells you which physics terms to add, and the honest
version of the paper reports the gap.

**Exit:** validation figure and table in `results/`, written up in `docs/decisions.md`.

---

### Phase 6 — MoCo-v2 pretraining

`src/phase/ssl/train_moco.py`. Config-driven, three variants:
`moco_generic`, `moco_physics`, `moco_physics_demon`.

Settings: ResNet-18 backbone, queue 4096–16384, momentum 0.999, temperature 0.2, 2-layer
MLP projection head to 128-D ℓ2-normalised, SGD momentum 0.9 weight decay 1e-4, cosine
schedule with 5-epoch warmup, 200 epochs, batch 128–256. Single GPU: replace shuffling-BN
with SplitBN or LayerNorm.

DEMON-consistency term: auxiliary loss penalising divergence between the DEMON spectrum
implied by the two views' embeddings. Weight is a swept hyperparameter.

Health monitors, logged every epoch — these are how you know SSL is working before you
have any downstream number:

- k-NN probe accuracy on validation.
- Alignment: `E‖f(x) − f(x⁺)‖²`.
- Uniformity: `log E e^{−2‖f(x) − f(y)‖²}`.
- Collapse detection: per-dimension embedding std, rank of the embedding covariance.

If uniformity plateaus high while k-NN sits at chance, the representation has collapsed —
usually augmentation too weak. Fix before burning more compute.

**Exit:** three pretrained checkpoints, monitor curves logged, no collapse.

---

### Phase 7 — Evaluation

**Protocols:** linear probe on frozen encoder, and full fine-tune. Report both.

**Label-efficiency curves** at 1%, 5%, 10%, 25%, 50%, 100% of labels. Subsample by
recording, stratified by class. This is the headline figure.

**Cross-dataset:** train on DeepShip → test on ShipsEar, and reverse. Expect degradation;
the claim is that PHASE degrades less.

**Baselines, all mandatory:**

1. Random-init supervised
2. Supervised-only at 100% labels
3. ImageNet-pretrained transfer
4. AudioSet-pretrained transfer (PANN or AST)
5. Generic-augmentation MoCo (this is the direct ablation of the whole idea)
6. Recording-ID shuffle control

**Metrics:** accuracy, macro-F1, per-class F1, confusion matrix, AUC. Three seeds, mean ±
std, paired t-test against baseline 5.

**Ablations:**

| Ablation | Variants |
|---|---|
| Augmentation family | generic → +Lloyd's Mirror → +multipath → +TL → +noise → full |
| DEMON loss | off / on, weight sweep |
| Queue size | 1024 / 4096 / 16384 |
| Temperature | 0.07 / 0.2 / 0.5 |
| Feature | log-Mel / CQT / dual-stream with DEMON |
| Backbone | ResNet-18 / ResNet-34 / AudioSet-AST transfer |

**Exit:** results tables in `results/tables/`, every number three-seeded, leakage control
reported alongside.

---

### Phase 8 — Interpretability and report

- t-SNE and UMAP of embeddings, coloured by class and separately by recording.
- Grad-CAM over spectrograms — does the model attend to the tonal lines?
- Attention maps if the AST transfer baseline is included.

**Report structure:** Abstract · Introduction · Related Work · Method · Datasets and
Preprocessing · Experiments · Results and Analysis · Interpretability · Limitations ·
Conclusion.

**Required figures:** architecture diagram; augmented spectrograms with preserved DEMON
peaks; label-efficiency curves (headline); DS3500 physics-validation comparison; t-SNE by
class vs by recording; confusion matrices; alignment/uniformity/k-NN training curves;
Grad-CAM.

**Required tables:** dataset summary; main results with macro-F1 ± std against all
baselines; ablation table; cross-dataset matrix.

---

## 4. Twelve-week schedule

| Week | Phase | Deliverable |
|---|---|---|
| 1 | 0, 1 | Repo scaffolded, all four datasets downloaded, DeepShip audit and decision recorded |
| 2 | 2, 3 | Manifests, immutable leakage-safe splits, leakage test green, ESC-50 dry run passes |
| 3 | 4 | Physics augmentation library complete, DEMON-preservation test green |
| 4 | 5 | DS3500 physics validation figure and table |
| 5 | 6 | Supervised DeepShip baseline reproduces published range |
| 6–7 | 6 | Three MoCo variants pretrained, monitors clean |
| 8 | 7 | Linear probe and label-efficiency curves |
| 9 | 7 | Cross-dataset, all baselines, leakage control |
| 10 | 7 | Ablations, three seeds, significance tests |
| 11 | 8 | Interpretability figures |
| 12 | 8 | Report written and polished |

---

## 5. Risks and fallbacks

| Risk | Trigger | Fallback |
|---|---|---|
| ~~DeepShip mirror unusable~~ | Resolved | Gate passed on `tangqiji/deepship-raw`, 47.22 h. D-007 |
| SSL does not beat supervised at 100% labels | Phase 7 | Expected. Pivot narrative to label efficiency and cross-dataset robustness, where SSL usually still wins. Report honestly |
| Representation collapse | Phase 6 monitors | Strengthen augmentation, raise queue size, lower temperature |
| Physics augmenter diverges badly from BELLHOP | Phase 5 | Report the gap as a finding, add multipath arrivals, reduce claim scope to shallow-water regime |
| Colab session limits | Any | Per-epoch checkpointing to Drive, resume-from-checkpoint in every training script |
| Out of time | Week 10 | Minimum viable result: generic-vs-physics MoCo label-efficiency curve on DeepShip, recording-safe splits, three seeds. Everything else is optional |

---

## 6. First command to run

Start with Phase 0. Scaffold the repository exactly as specified in section 1, set up
`pyproject.toml`, `seed.py`, `config.py`, and the four test stubs. Then move to Phase 1
and run the DeepShip audit. Do not begin Phase 2 until the decision gate is recorded in
`docs/decisions.md`.
