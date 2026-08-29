# Decisions

Every decision that departs from `PLAN.md`, or that a later phase depends on, is recorded
here with its evidence.

---

## D-001 — Local development runs in a project venv, not the global interpreter

**Date:** 2026-08-27 · **Phase:** 0

The global Python 3.13 interpreter carries a 4.3 GB `torch` directory in
`site-packages` with no `dist-info`. `pip list` does not see it and its DLLs fail to load
with `WinError 193`, the signature of an interrupted install. It shadows every
`import torch`.

Rather than repair or delete a 4.3 GB directory shared with the user's other projects, the
project installs into `.venv/` at the repository root. The global interpreter is left
untouched. `.venv/` is gitignored.

Local torch is the CPU build (2.13.0+cpu). It is there so the test suite and the feature
and augmentation code can be exercised without a GPU. Training runs on Colab or Kaggle.

---

## D-002 — torch and torchaudio are range-pinned, everything else is exact-pinned

**Date:** 2026-08-27 · **Phase:** 0

`PLAN.md` §Phase 0 asks for pinned versions. Every dependency is pinned with `==` except
torch and torchaudio, which are `>=2.4,<3`.

Colab and Kaggle ship a preinstalled torch built against the runtime's CUDA driver.
`pip install torch==X` on those platforms either triggers a multi-gigabyte reinstall or
installs a build whose CUDA version does not match the driver. `scripts/00_setup_env.sh`
detects a working torch and leaves the platform build in place.

---

## D-003 — Test stubs skip rather than fail

**Date:** 2026-08-27 · **Phase:** 0

`PLAN.md` §Phase 0 asks for the four tests as failing stubs. They are written as skipping
stubs instead: `test_splits_no_leakage.py` and `test_manifest_integrity.py` are fully
implemented but parametrised over artefacts that do not exist yet, so they skip and turn
green the moment Phase 2 writes a split or manifest file. `test_augment_preserves_demon.py`
skips at module level with a Phase 4 reason.

A suite that is red from day one stops being a signal. Skips with explicit reasons stay
visible in the pytest summary and cannot be mistaken for passing coverage.

`test_config_roundtrip.py` is implemented in full now, because Phase 0 is the phase that
makes it meaningful.

---

## D-004 — Two files added outside the `PLAN.md` §1 layout

**Date:** 2026-08-27 · **Phase:** 0

- `src/phase/tracking.py` — W&B run initialisation. `CLAUDE.md` requires every run to log
  its full config, git SHA, and seed. No module in the §1 layout owns that;
  `ssl/monitors.py` is for SSL health metrics, not run setup.
- `scripts/99_smoke_wandb.py` — the Phase 0 W&B exit criterion. Kept out of
  `00_setup_env.sh` because it needs an interactive `wandb login` or an explicit
  `--mode offline`.

No new top-level directories were created. `.venv/` is the only addition to the root and
it is gitignored.

---

## D-005 — Config validation encodes the `CLAUDE.md` correctness rules

**Date:** 2026-08-27 · **Phase:** 0

Rules that are easy to violate silently are enforced at config load rather than left to
discipline:

- `EvalConfig` rejects fewer than three seeds (`CLAUDE.md` rule 4).
- `AugmentOp` rejects `pitch_shift`, `time_warp`, `freq_mask_tonal`, and
  `spec_augment_freq` outright — the identity-destroying transforms named in Phase 4.
- `DatasetConfig` carries `split_safe`; DS3500 is `false` and every downstream consumer
  must check it before computing a headline number (`CLAUDE.md` rule 5).
- `PretrainConfig` rejects a MoCo queue size that is not a multiple of the batch size.
- `EvalConfig.leakage_control` defaults to `true` (`CLAUDE.md` rule 2).

---

## D-006 — ShipsEar/DS3500 audit: `CLAUDE.md` rule 5 is wrong about the mechanism, right
about the conclusion

**Date:** 2026-08-27 · **Phase:** 1 · **Evidence:** `data/manifests/shipsear_audit.json`,
`data/manifests/ds3500_audit.json`

`CLAUDE.md` rule 5 states: *"DS3500 has no recording IDs. Filenames encode class and
simulated position only."* The first sentence is right in effect. The second is factually
wrong on this release, and the difference matters for Phase 5.

**What the filenames actually encode.** Every file in both halves is named
`{class}_{group}_{segment}.wav`. Position is nowhere in the filename; it lives in
`DS3500/train_list.txt` and `DS3500/test_list.txt` as explicit range and depth columns.

```
ShipsEar/shipsear_5s_16k/0/0_0/0_0_2.wav      class 0, group 0, segment 2
DS3500/0_0_2.wav                              same clip, rendered
train_list.txt: ...0_0_2.wav   0   3.000   0.100
```

**The two halves are exact 1:1 matched pairs.** 2223 files each, and the set of filename
stems is identical between them. Every rendered clip has its dry original under the same
stem, and a labelled `(range, depth)` for the rendering.

**Consequences, in order of importance.**

1. **Phase 5 is easier and stronger than planned.** `PLAN.md` §Phase 5 asks to find
   "matched geometries" and compare. No matching is needed: the pairing is exact and
   per-clip, and all 36 geometries (range 1–11 km step 2, depth 0.1–1.1 km step 0.2) are
   present with 60–65 clips each. The analytic augmenter can be evaluated against BELLHOP
   clip by clip at a known geometry.
2. **`PLAN.md` overstates the rendering.** It says the 1,948 signals are rendered "at
   thirty-six simulated geometries", implying 1948 × 36. The release renders each clip at
   **one** assigned geometry, keeping the total at parity with the original — the dataset
   README says so explicitly. Position varies by segment index, cycling through the 36
   positions across each class. Phase 5 gets 2223 matched pairs spread over 36 geometries,
   not 70,128.
3. **Rule 5's conclusion stands, for a different reason.** Recording identity is not
   absent — there are 12 class-pure groups. It is *too coarse to split*: classes 2 and 4
   have exactly one group each, so no grouped split can place a group in train, val and
   test. `split_safe: false` is correct and stays. The audit reports this as
   `too_coarse_to_split` rather than `not_recoverable`, which is the honest description.
4. **The mirror is larger than published.** 2223 files against the documented 1948, in
   every class: observed 369/301/843/486/224 against published 345/235/785/395/188. The
   list files agree with the observed count (1778 train + 445 test = 2223), so the release
   is internally consistent and the paper's table is what disagrees. Any comparison to
   published ShipsEar numbers must state this.
5. **Class labels are `0`–`4`, not `A`–`E`.** The dataset README maps them in order, so
   `0`=A … `4`=E. Cosmetic, but the audit flags it as a spec miss and it must be applied
   when reading class names.

**Not changed:** `configs/data/shipsear_ds3500.yaml` keeps `split_safe: false`. No
headline label-efficiency number will be computed on this dataset.

---

## D-007 — DeepShip gate: the mirror `PLAN.md` names is a 3.4% subset, so the source moves

**Date:** 2026-08-27 · **Phase:** 1 · **Evidence:**
`data/manifests/deepship_subset_audit.json`

`PLAN.md` §2 names `vasundharauppuluri/deepship-main` and flags it "Unverified mirror.
Phase 1 gate." The gate caught it.

**What that mirror actually contains.**

| | Published DeepShip | `deepship-main` mirror |
|---|---|---|
| Hours | 47 h 04 min | **1.59 h** (3.4%) |
| Ships | 265 | 45 distinct names |
| Files | — | 63 |
| Classes | 4 | 4 (Cargo 12, Passenger 20, Tanker 28, **Tug 3**) |
| Sample rate | 32 kHz | 32 kHz mono |

Its own `README.txt` says so: *"keeping in view the limitation of file size and space on
github directory, a part of dataset could be uploaded on this repository. The remaining
part can be downloaded by sending the email to mirfan@mail.nwpu.edu.cn"*.

**Recording identity is recoverable there, but the scale is not usable.** Each class ships
a `*-metafile` with `index, type_code, SHIP NAME, date, time, duration`, and all 63 wavs
join to it by filename index. That is `PLAN.md`'s "subset but per-transit boundaries
survive" branch on the letter of it. It fails on the substance: with three tug recordings,
a grouped train/val/test split puts **one tug recording in test**, and the 1%-label point
on the headline label-efficiency curve is less than one file. The curve is not computable.

Two further defects: only 23 of 63 durations agree with the metafile (several wavs are
truncated — 199 s against a stated 249 s), and the structural audit alone returns
`not_recoverable`, because identity lives in a sidecar file rather than in the paths.

**Decision: move the source to `tangqiji/deepship-raw`.** 15.49 GB, 296 downloads, and the
structure is better than the subset's metafile join:

```
Cargo/20171104-1/1.wav
Cargo/20171104a-2/2.wav
Cargo/20171105a-3/3.wav
```

`{date}-{transit}/` is a per-transit directory, so `recording_id` comes straight from the
path and `splits.py` needs no sidecar. 47 h at 32 kHz mono is about 10.8 GB, so 15.49 GB
is the right order for the complete set. `deku14all1/deepship` is byte-identical for the
first 20 files and serves as a backup mirror.

`configs/data/deepship.yaml` now points at `kaggle:tangqiji/deepship-raw`. The subset is
kept at `data/raw/deepship_subset_vasundharauppuluri/` as gate evidence.

**Closed 2026-08-28. The swap is confirmed.** `data/manifests/deepship_audit.json`:

| Check | Expected | Observed |
|---|---|---|
| Hours | 47.07 | **47.22** |
| Classes | cargo, passenger, tanker, tug | Cargo, Passenger, Tanker, Tug |
| Sample rate | 32 kHz | 32 kHz mono, 32-bit float |
| Recordings | 265 ships | **609 transits** |

**4/4 published-spec checks pass.** 609 files, one per transit directory, Tug 69.

One incident on the way: the first extraction stopped at 542 of 609 files, taking almost
all of Tug with it (2 of 69 survived), and `extract_archives` could not see it because its
guard treated *any* audio under the target as proof of a complete extraction. Fixed in
D-012; the missing 68 members were recovered without re-downloading.

---

## D-008 — The Kaggle SDK buffers downloads in memory; `download.py` streams instead

**Date:** 2026-08-27 · **Phase:** 1

`KaggleApi.dataset_download_files` was observed growing resident memory at roughly 1 MB/s
while writing **zero** bytes to disk. The traceback from an interrupted run puts the
allocation in `requests.sessions.resolve_redirects` calling `r.content`, so a redirect hop
is being drained into memory rather than streamed. Two runs reached 1.2 GB resident with
nothing on disk. On a 15.49 GB dataset this exhausts memory before it produces a file.

`fetch_kaggle` no longer uses the SDK. It resolves redirects itself with
`allow_redirects=False`, streams 1 MB chunks straight to disk, and adds what a
multi-hour download on a throttled link needs:

- **Resume** via a `Range` header from the existing file size, verified by stopping a
  download at 286 MB and restarting it at 369 MB.
- **Retry** with exponential backoff over 20 attempts, resuming from disk each time.
- **Short-read detection** so a truncated transfer raises instead of being extracted.

Credentials come from `KAGGLE_USERNAME`/`KAGGLE_KEY` or `~/.kaggle/kaggle.json`, and are
never logged.

---

## D-009 — Full DeepShip download is bandwidth-blocked

**Date:** 2026-08-27 · **Phase:** 1 · **Status:** in progress

Kaggle is serving this account at **0.3–0.5 MB/s** sustained. The same link pulled DS3500
from HuggingFace at about 7 MB/s, so the limit is Kaggle-side, not local. At that rate the
15.49 GB mirror needs roughly **10 hours**.

An early SDK run appeared to fetch 503 MB at 237 MB/s. Re-measuring the same dataset with
the streaming fetcher gave 0.5 MB/s, so that figure was a warm cache, not the real link.

HuggingFace has no DeepShip mirror — its dataset search returns nothing for `deepship`,
`shipsear`, or `ship radiated noise`.

The download is resumable and safe to stop and restart. It must finish before the
confirming audit in D-007 can be run and before Phase 2 starts.

---

## D-010 — Concurrent downloads silently corrupted an archive; the fetcher now guards against it

**Date:** 2026-08-27 · **Phase:** 1

An ESC-50 fetch produced a **2,872,799,302-byte** zip where the source is
**1,528,150,051 bytes**, and `zipfile.testzip()` reported the first entry corrupt. The
extra bytes were a second full download appended onto an existing partial file: two
processes writing the same path, each resuming from a size the other was changing.

Response headers ruled out the alternative explanations — the GCS hop reports
`x-goog-stored-content-encoding: identity`, `Content-Length: 1528150051` and
`Accept-Ranges: bytes`, so neither transfer compression nor a mis-parsed range was
involved.

This was found on a 1.5 GB dataset. On the 15.49 GB DeepShip archive the same failure
would surface only after hours, as a corrupt extraction rather than an obvious error.

`download.py` now:

- **Serialises writers.** A `.lock` file beside the target, created `O_EXCL`, refuses a
  second download of the same archive. The holder touches it every two seconds, so a lock
  older than 120 seconds is treated as stale and cleared automatically — a killed process
  cannot block the next run forever.
- **Downloads to `.part` and renames on success.** A partial transfer can never be
  mistaken for a finished archive.
- **Verifies size and MD5.** The final size must equal `Content-Length`, and the content
  hash is checked against the `md5` component of `x-goog-hash` when the server sends it.
  A mismatch deletes the file rather than leaving it to be extracted.
- **Rejects a local file larger than the source**, discarding and restarting instead of
  resuming from a nonsensical offset.
- **Validates archives before extracting.** `extract_archives` runs `testzip()` first, and
  deletes a corrupt archive with an explicit message instead of writing a partial tree.

MD5 verification only applies to a download that ran start-to-finish in one attempt; a
resumed transfer has no hash for the bytes already on disk, so it falls back to the size
check plus `testzip()` at extraction.

---

## D-011 — QiandaoEar22 is unobtainable; ambient noise is re-sourced, not dropped

**Date:** 2026-08-28 · **Phase:** 1

QiandaoEar22 sits behind an IEEE DataPort membership the project does not have. Its three
roles in `PLAN.md` are not equally affected:

1. **Fallback primary** if the DeepShip gate failed — moot. The gate passed with the full
   47.22 h.
2. **Extra unlabelled pretraining data** — moot for the same reason.
3. **Ambient noise source for the `noise_mixing` physics augmentation** — the only live
   role, and the one worth preserving. Dropping it would cost PHASE one of its five
   physics transforms and shorten the augmentation-family ablation from six rungs to five.

**Replacement, in two parts:**

- **ShipsEar class E** — 224 clips of real recorded underwater environmental noise, at
  16 kHz, already on disk. DS3500 additionally supplies BELLHOP-rendered versions of the
  same 224 clips, so the noise bank itself can be propagated.
- **Wenz-curve sea-state model** — parametric ocean ambient from the standard
  shipping-plus-wind-plus-thermal decomposition. Unlimited quantity, sea state is a
  controllable parameter, and it arguably suits a physics-augmented method better than
  sampling a freshwater lake dataset recorded in a different basin.

**Contamination guard, enforced in code rather than by discipline.** Cross-dataset
evaluation (DeepShip → ShipsEar) is a headline result, so ShipsEar class-E clips used as
mixing noise must never appear in any ShipsEar evaluation split. Noise clips are drawn only
from class-E recordings held out of every ShipsEar split, and Wenz-synthetic noise is
preferred whenever ShipsEar is the evaluation target.

`configs/data/qiandaoear.yaml` is deleted and `PLAN.md` §2 and §5 are updated to match.

---

## D-012 — `extract_archives` treated a partial extraction as complete

**Date:** 2026-08-28 · **Phase:** 1

The DeepShip extraction stopped at 542 of 609 wavs. Because members are written in archive
order and Tug sorts last, the loss fell almost entirely on one class: **2 of 69 Tug
recordings survived**. The audit reported 36.35 h against a published 47.07 and a Tug class
too thin to split, which reads exactly like a bad mirror rather than a bad extraction.

The guard was `if target.is_dir() and count_audio(target): continue` — *any* audio under
the target counted as done, so re-running the fetch skipped the incomplete tree silently
and permanently.

`extract_archives` now compares the archive's own member list against what is on disk,
matching each member by name **and file size**, and extracts only what is missing or
truncated. It re-checks afterwards and raises if anything is still absent. `fetch()` no
longer short-circuits before extraction when audio already exists, so a partial tree is
repaired on the next run instead of being skipped.

Recovering the 68 missing members took seconds and needed no re-download. Had this gone
unnoticed, every downstream result would have been computed on a Tug class of two
recordings.

---

## D-013 — `recording_id` groups by ship, not by transit

**Date:** 2026-08-28 · **Phase:** 2

`PLAN.md` §2 defines `recording_id` as the ship transit. The metafiles show why that is too
weak: the same vessel is recorded on many dates.

| Class | Transits | Distinct ships | Busiest vessel |
|---|---|---|---|
| Cargo | 109 | 62 | PRINCESS SUPERIOR, 20 transits |
| Passenger | 191 | 43 | QUEEN OF, 35 transits |
| Tanker | 240 | 120 | KIRKEHOLMEN, 3 transits |
| Tug | 69 | 16 | SEASPAN RAVEN, 16 transits |

Grouping by transit would place 35 recordings of QUEEN OF across train, val and test.
Two transits of one hull share a propeller and machinery signature, so that is
ship-identity leakage even though no single recording straddles a split. Grouping by ship
is strictly stronger and remains feasible: the thinnest class still has 16 distinct
vessels, well above the three needed for a three-way split.

**Join.** Ship names come from the metafiles, which are split across mirrors — the full
mirror carries Cargo and Passenger, the retained subset carries Tanker and Tug. Matching a
transit directory to a metafile row uses `(date, time)` first, falling back to
`(date, index)`; between them **586 of 609 transits (96.2%)** resolve to a ship name. The
23 unmatched transits are all Tanker and fall back to transit-level grouping, each its own
group. The manifest records which rule matched, and the count is reported rather than
buried.
---

## D-014 — DeepShip carries a systematic DC offset; the corpus is cleaned and re-encoded

**Date:** 2026-08-28 · **Phase:** 3 · **Evidence:**
`data/manifests/deepship_clean_report.json`

Measured across all 609 files: DC ranges from **-0.03179 to +0.03434**, against an AC RMS
of **0.00063 to 0.07038**. On typical files the offset is **3-17x larger than the signal
itself**. ShipsEar (DC ~2e-5) and ESC-50 (DC ~1e-5) are clean, so this is DeepShip-specific.

This is not cosmetic. DEMON detects the envelope and then FFTs it; a pedestal an order of
magnitude above the signal would dominate that envelope, and the shaft and blade lines are
the cue the entire method exists to protect. The Phase 6 DEMON-consistency loss would have
been built on it silently.

`src/phase/data/clean.py` runs two passes: measure, then write. Output is DC-removed,
gain-normalised, 16-bit FLAC at 32 kHz under `data/clean/deepship/`, mirroring the
`{class}/{date}-{index}/{time}.flac` layout so `recording_id` and the frozen splits still
resolve. The corpus drops from **21.7 GB to roughly 7.7 GB**, which fits free Drive.

Quality outliers are **flagged, not dropped**: 0 dead channels, **57 files** with more than
90% of their AC energy below 10 Hz (drift rather than ship noise), and **50 files with
negative DC**, possibly polarity-inverted. Excluding any of them would be a separate
recorded decision.

---

## D-015 — Per-file gain, not one global gain

**Date:** 2026-08-28 · **Phase:** 3

The Phase 3 plan specified a single global gain, on the stated grounds that
transmission-loss augmentation and SNR-controlled noise mixing depend on relative loudness
between recordings. **That reasoning was wrong and is corrected here.** Both operate
window-relative — transmission loss rescales a window against itself, and noise mixing sets
SNR from the window's own RMS — and `PLAN.md` §Phase 3 specifies per-sample z-score
normalisation at feature time, which discards absolute level regardless.

The cost of the global gain was real. Per-file AC peaks span **0.0039 to 1.030, a 264:1
range**, so one gain set by the loudest file left the quietest at **6.9 effective bits**
and a 25.6 dB quantisation SNR; 200 files (32.8%) fell below 10 effective bits.

Each file is now scaled to 0.95 of full scale by its own AC peak, and **the gain is
recorded per file in the clean report**, so absolute level is not lost — it moves from the
samples into metadata and can be undone by any consumer that needs it.

The cleaned corpus is larger than under the global gain (7.7 GB against 4.1 GB) precisely
because more information is retained for FLAC to encode.

---

## D-016 — No waveform cache; windows are an index, read on demand

**Date:** 2026-08-28 · **Phase:** 3

Measured rather than assumed: random 30 s window reads through `soundfile` seeking run at
**13.5 ms each, 74 windows/s single-threaded**. One epoch over 10,479 DeepShip windows is
142 s of pure I/O on one thread, which a few DataLoader workers overlap with GPU compute.

| Option | Size |
|---|---|
| float32 waveform cache | 40.2 GB |
| float16 / int16 waveform cache | 20.1 GB |
| log-Mel cache, 128 mels, hop 512, fp16 | 5.0 GB |
| **cleaned FLAC, read on demand** | **7.7 GB** |

The decisive point is not size. **MoCo needs waveforms at training time**: Lloyd's Mirror is
a comb filter, multipath is delay-and-sum, Doppler is resampling. None can be applied to a
precomputed log-Mel, so a feature cache cannot serve the pretraining path at all.

`src/phase/data/segment.py` therefore emits an index of
`(path, start, frames, class, recording_id, split)` rather than audio files, and
`data/cache/` holds features only for paths where augmentation is off — the ESC-50 dry run
and linear probes on frozen features.

---

## D-017 — The configured DEMON band exceeds Nyquist and is clamped

**Date:** 2026-08-28 · **Phase:** 3

`PLAN.md` §Phase 3 specifies a **1-30 kHz** DEMON analysis band, and
`configs/features/demon.yaml` encodes it. That band is unreachable: DeepShip is sampled at
32 kHz, so Nyquist is 16 kHz, and ShipsEar at 16 kHz gives only 8 kHz.

`src/phase/features/demon.py` clamps the upper edge to 0.98 of Nyquist per dataset — 15.68
kHz for DeepShip, 7.84 kHz for ShipsEar — and lowers the bottom edge if it would exceed
half the top. The configured value is kept as the requested band so the intent stays
visible in the config.

Validated on a synthetic carrier with a known 6.20 Hz shaft rate and 4 blades: the
implementation recovers peaks at **6.35 Hz and 24.9 Hz** against injected 6.20 and 24.80 Hz.

---

## D-018 — Augmentations are batched torch ops, not per-window NumPy

**Date:** 2026-08-28 · **Phase:** 4

`PLAN.md` §Phase 4 describes each augmentation as "a callable taking a waveform and a
sampled parameter dict, returning a waveform" — per-window NumPy in the DataLoader. Measured
before building it, that costs **190 ms per 30 s window** for Lloyd's Mirror alone: a
full-window rFFT/irFFT over 960,000 samples at 32 kHz. Two views at batch 128 is roughly
48 s per batch for one transform, before any GPU work. Over 200 epochs it is not runnable.

The same operation batched in torch costs **14.2 ms per window on CPU**, a 13x speedup, and
is device-agnostic so it moves to the GPU unchanged.

The interface keeps the spirit of `PLAN.md` — one class per transform, parameters drawn
per view, geometry inspectable via `sample_params` — but operates on a `(batch, samples)`
tensor and is applied in the training step rather than the DataLoader.

`src/phase/augment/base.py` adds `SpectralAugmentation`, whose subclasses return a per-item
frequency response instead of a waveform. `Pipeline` detects consecutive spectral ops and
applies them through **one** rFFT/irFFT pair: Lloyd's Mirror, multipath and transmission
loss now share a single transform pair rather than paying for three.

---

## D-019 — Two corrections to the physics in `PLAN.md`

**Date:** 2026-08-28 · **Phase:** 4

**The Lloyd's Mirror sign.** `PLAN.md` gives `|H(f)|² = 2[1 + cos(2πfΔτ)]`. The sea surface
is pressure-release, with a reflection coefficient of **−1** — which
`configs/augment/physics.yaml` already encoded before this was noticed — so the correct
form is `2[1 − cos(2πfΔτ)]`, a null at DC rather than a peak. The implementation reads the
coefficient from config, so both are expressible, and defaults to the physical −1.

**Filters are complex, not magnitude-only.** The first implementation applied
`|H(f)|` as a zero-phase filter. That is wrong for these transforms: multipath *is* phase —
delay-and-sum is `Σ gain_k · e^{−j2πfτ_k}` — and a zero-phase version reproduces the
magnitude comb while losing the temporal smearing of the envelope, which is exactly the
effect DEMON-consistency has to be robust to. All three spectral transforms now carry true
complex responses. This also makes multipath exact for fractional delays, where the previous
integer-sample `gather` rounded.

Complex arithmetic costs roughly 2-3x the magnitude-only version on CPU. It is kept because
it is correct; see D-020.

---

## D-020 — Augmentation cost is a GPU-verified open risk

**Date:** 2026-08-28 · **Phase:** 4 · **Status:** open, must close before Phase 6

Measured on CPU for the full seven-op physics pipeline, both views, 30 s windows:

| Op | ms per view-window |
|---|---|
| lloyds_mirror | 52.5 |
| multipath | 66.3 |
| transmission_loss | 22.7 |
| noise_mixing | 53.4 |
| doppler | 11.3 |
| time_mask | 4.7 |
| **full pipeline** | **~100-170** |

At batch 128 that is roughly 40 s per batch on CPU. Not runnable.

**The target is a GPU and these are elementwise complex ops over a few million elements —
precisely what a GPU is for — but the speedup is an expectation, not a measurement.** Local
torch is CPU-only, so it could not be measured here.

Section 8 of `notebooks/colab_eval.ipynb` benchmarks the pipeline at batch 8, 32 and 128 on
a T4. **Phase 6 must not start until that number is known.** If it is still too slow, the
fallbacks in order of preference are: shorter windows (10 s measured at 3.7 ms per window
against 30 s at 14.2 ms), applying the spectral chain at a reduced sample rate for the
log-Mel path only, or precomputing a bank of transfer functions and sampling from it.

Recording the honest position: the 13x batching speedup in D-018 is real and measured, but
it was measured on a single magnitude-only op, and the full correct pipeline is
substantially more expensive than that figure implies on its own.

---

## D-021 — Phase 5 negative result: the analytic channel does not approximate BELLHOP

**Date:** 2026-08-28 · **Phase:** 5 · **Evidence:**
`results/tables/physics_validation.json`, `results/figures/physics_validation.png`

216 matched pairs, six per geometry across all 36, each rendered at its **actual** labelled
range and receiver depth rather than a sampled one. Metric is log-spectral distance to the
BELLHOP rendering; the untouched ShipsEar original is the baseline.

| Variant | LSD dB | 10-100 Hz | 100-500 | 500-2k | 2k-8k | DEMON cos |
|---|---|---|---|---|---|---|
| **none (baseline)** | **6.348** | **6.874** | 6.449 | 6.199 | 6.283 | 0.894 |
| lloyd | 9.786 | **17.483** | 9.218 | 9.142 | 9.597 | 0.879 |
| tl | 6.525 | 6.764 | 6.891 | 6.375 | 6.436 | 0.891 |
| lloyd+tl | 9.823 | 15.782 | 9.017 | 9.376 | 9.678 | 0.878 |
| lloyd+tl+multipath | 10.420 | 15.533 | 9.801 | 9.996 | 10.312 | 0.869 |

**Applying the analytic channel makes agreement worse, not better.** Lloyd's Mirror costs
about **+3.4 dB**, and the damage is concentrated below 100 Hz where it adds **+10.6 dB**.
The penalty is flat across receiver depth (+3.2 to +3.6 dB at every depth), so it is not a
tuning problem. Transmission loss alone is roughly neutral (+0.18 dB). Multipath adds
further error. No variant beats doing nothing.

**Why, and it is not an implementation bug.** Lloyd's Mirror is a two-ray surface
interference model: direct path plus surface image, valid for a near-surface receiver in
shallow, effectively isovelocity water. DS3500 renders a **3500 m deep-sea** environment
with a WOA18 sound-speed profile, and its own README says it models "direct and shadow
zones". That regime is refraction-dominated — rays bend, shadow zones open, and the deep
sound channel matters. A surface-image model has no way to represent any of it. The comb
structure the implementation produces is correct (verified in
`results/figures/augmentations.png`, notches at 1.9/3.7/5.6/7.4 kHz for the sampled
geometry); it is the right filter for the wrong ocean.

**The geometries do not overlap, which bounds what this test can say.**

| | `configs/augment/physics.yaml` | DS3500 |
|---|---|---|
| receiver depth | 10-60 m | **100-1100 m** |
| range | 200-6000 m | 1000-11000 m |

The augmenter's configured receiver depth and DS3500's have **no overlap at all**. This test
therefore measures the model far outside its intended envelope, and cannot speak to the
shallow, near-surface regime it was written for. DS3500 offers no rendering there, so that
regime remains unvalidated rather than validated.

**What this does and does not invalidate.**

- It **does** falsify the specific claim in `PLAN.md` §Phase 5 that "a computationally cheap
  analytic channel model yields representations comparable to full ray-model simulation" —
  at least for deep water. That sentence must come out of the abstract.
- It **does not** touch the central label-efficiency claim. The augmentations do not need to
  reproduce BELLHOP; they need to generate plausible channel variation so the encoder learns
  invariance. Matching a specific simulator is a stronger property than the method requires.
- DeepShip is recorded in the Strait of Georgia, coastal water of order 100-400 m — much
  closer to Lloyd's Mirror's regime of validity than DS3500's 3500 m, though still deeper
  than the config samples.

**A caveat on the metric, stated so it is not discovered later.** Pointwise log-spectral
distance at matched geometry asks whether the augmenter *reproduces* one rendering. The
augmenter is stochastic and is meant to span a distribution of channels, so the fairer
question is whether the BELLHOP rendering lies within the span of what it generates. That is
a distributional test, not a distance, and it is not what `PLAN.md` asked for. The pointwise
result above is clear enough that a distributional test is unlikely to reverse it, but it
would be the more appropriate framing for a paper.
