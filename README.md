# Reproducibility materials — Submission 1027

This public artifact contains the source-only clean-20260811 implementation
associated with submitted manuscript 1027. It is a submission artifact, not an
accepted-paper record. EMG is privileged information during training; Stage 3
deployment and inference use EEG only. The repository contains no T-MSPD data,
feature arrays, checkpoints, trial predictions, raw result files, or audio
cache.

The work is a controlled comparison under one prespecified protocol. It does
not claim state of the art; Full outperforms the included EEGNet-8,2 baseline
under the reported protocol.

## Method

1. **Stage 1, EMG teacher.** A masked autoencoder learns from raw EMG, with
   lexical classification and Overt-only acoustic alignment. Acoustic targets
   are fixed HuBERT Base frame features from the **ninth Transformer block**.
   The 768-D frames are temporally mean-pooled per trial and passed through a
   `768-512-256` projection MLP. `AttentionPooling` is not part of the final
   pipeline. Stage 1 produces a frozen EMG teacher; it does not initialize any
   trainable EEG parameter.
2. **Stage 2, EEG pretraining.** An EEG masked autoencoder trains for 150
   epochs. Epochs 1-30 use reconstruction only. Thereafter the configured
   representation loss and `0.1 * cross_entropy` are added. Checkpoints are
   selected only by pooled validation balanced accuracy. In distillation
   conditions, EMG generates the target while EEG enters the student loss.
3. **Stage 3, EEG-only fine-tuning.** A subject-dependent classifier is trained
   for 100 epochs on the clean `outer_seed_20260811` five-fold split. The
   validation-selected checkpoint is reloaded before the final-test loader is
   constructed and evaluated exactly once.

Stage-2 conditions are:

- `full`: paired EEG-EMG target plus the same-class prototype term, `alpha=0.3`.
- `no_distill`: reconstruction and auxiliary classification, without the
  representation loss.
- `unrestricted_shuffle`: EMG donors are deranged across the training
  mini-batch, which breaks trial and class identity.
- `within_class_shuffle`: a one-to-one no-fixed-point EMG derangement within
  subject, paradigm, and lexical class, on the training partition only.
- `instance_only_alpha0`: exact paired EMG target with `alpha=0`; the
  class-prototype term is absent and every other Full setting is unchanged.
- `label_simplex`: no EMG data or checkpoint is opened. The lexical label
  selects one row of a frozen 10-class regular simplex embedded in 256
  dimensions, with representation-loss weight `1.0` after the same warm-up.

## Data and acoustic features

T-MSPD is restricted access. Apply through the
[official dataset page](https://cstr.cn/31253.11.sciencedb.24416), download it
under its terms, and keep it outside this repository. See [docs/DATA.md](docs/DATA.md).

Users must independently generate the HuBERT cache. This repository does not
distribute raw audio, HuBERT weights, extraction caches, or feature arrays. The
loader keeps the historical generic `*_audio_feat.npy` cache API for
compatibility. Overt cache entries have shape `(N, 50, 768)`.

The submitted method record identifies HuBERT Base and the ninth Transformer
block, but the historical cache-extraction script, exact model revision, and
audio sampling/resampling provenance are not included in the retained sources.
Consequently this repository validates and consumes the frozen cache contract;
it does not claim that a particular source sampling rate or resampling step was
historically executed. Archiving that extractor provenance remains pending.

Configure local paths at the command line or in a local copy of
`configs/paths.example.yaml`. No server path is embedded in the code.

The submission-to-source evidence map is in
[docs/IMPLEMENTATION_CROSSWALK.md](docs/IMPLEMENTATION_CROSSWALK.md). The
architecture, complete experiment grids, interpretation limits, and
reproducibility status are documented separately under `docs/`;
submission-only provenance is not promoted to a frozen-source claim.

## Environment

Python 3.10+ and a CUDA-compatible PyTorch installation are required for
training. The audited environment is documented in
[TESTED_ENVIRONMENT.md](TESTED_ENVIRONMENT.md).

```bash
pip install -e '.[analysis,visualization,test]'
```

## Reproduction order

Create the deterministic clean split once:

```bash
python scripts/prepare_splits.py \
  --data-root /path/to/authorized/T-MSPD \
  --output /path/to/generated/splits
```

Generate auditable task identities. The output directory must not exist:

```bash
python scripts/generate_protocol_manifests.py \
  --data-root /path/to/authorized/T-MSPD \
  --split-root /path/to/generated/splits \
  --run-root /path/to/new/runs \
  --output-dir /path/to/new/manifests
```

This emits both the compatibility manifests and the complete grids:
`stage2_six_variant_90.jsonl`, `stage3_six_variant_5400.jsonl`,
`stage2_clean_main_four_variant_60.jsonl`, and
`stage3_clean_main_four_variant_3600.jsonl`. Shared Stage 1 is 15 cells and is
counted once.

Run Stage 1, then each Stage-2 condition for seeds `42`, `2026`, and `3407`
and folds `0-4`. Example single cells:

```bash
python scripts/train_stage1.py --config configs/stage1.yaml \
  --data-root /path/to/authorized/T-MSPD --split-root /path/to/generated/splits \
  --seed 42 --fold 0 --epochs 150 --output /path/to/new/runs/stage1/seed42/fold0

python scripts/train_stage2.py --config configs/stage2_full.yaml \
  --data-root /path/to/authorized/T-MSPD --split-root /path/to/generated/splits \
  --seed 42 --fold 0 --epochs 150 --variant full \
  --stage1-checkpoint /path/to/stage1/best_stage1.pth \
  --output /path/to/new/runs/stage2/full/seed42/fold0

python scripts/train_stage2.py --config configs/stage2_instance_only_alpha0.yaml \
  --data-root /path/to/authorized/T-MSPD --split-root /path/to/generated/splits \
  --seed 42 --fold 0 --epochs 150 --variant instance_only_alpha0 \
  --stage1-checkpoint /path/to/stage1/best_stage1.pth \
  --output /path/to/new/runs/stage2/instance_only_alpha0/seed42/fold0

python scripts/train_stage2.py --config configs/stage2_label_simplex.yaml \
  --data-root /path/to/authorized/T-MSPD --split-root /path/to/generated/splits \
  --seed 42 --fold 0 --epochs 150 --variant label_simplex \
  --output /path/to/new/runs/stage2/label_simplex/seed42/fold0
```

Then run all 30 subjects, both paradigms, five folds, and three seeds:

```bash
python scripts/train_stage3.py --config configs/stage3.yaml \
  --data-root /path/to/authorized/T-MSPD --split-root /path/to/generated/splits \
  --seed 42 --fold 0 --epochs 100 --subject S01 --paradigm overt \
  --stage2-checkpoint /path/to/stage2/best_stage2.pth \
  --output /path/to/new/runs/stage3/full/seed42/fold0/S01/overt
```

For LabelSimplex Stage 3, use `configs/stage3_label_simplex.yaml`, pass
`--variant label_simplex`, and provide the corresponding validation-selected
LabelSimplex Stage-2 checkpoint. That path uses only EEG and label arrays.

Every output path must be new. A completed `result.json` is immutable; do not
rerun a successful task based on its final-test score.

## EEGNet and secondary analyses

The included EEGNet-8,2 implementation trains from scratch using EEG only:

```bash
python scripts/generate_eegnet_manifest.py \
  --data-root /path/to/authorized/T-MSPD --split-root /path/to/generated/splits \
  --run-root /path/to/new/eegnet-runs --output-dir /path/to/new/eegnet-manifest
python scripts/train_eegnet.py --manifest /path/to/new/eegnet-manifest/eegnet_3seed_900tasks.jsonl --index 0
python scripts/aggregate_eegnet.py --manifest /path/to/new/eegnet-manifest/eegnet_3seed_900tasks.jsonl --out-dir /path/to/new/eegnet-summary
```

The strict three-seed subject-isolated pipeline is separate from clean-main.
Generate its six-subject outer-fold exclusion grid and separate target split
role files, then run individual manifest tasks as follows:

```bash
python scripts/generate_subject_isolated_manifests.py \
  --source-split-root /path/to/generated/splits \
  --run-root /path/to/new/subject-isolated-runs \
  --output-dir /path/to/new/subject-isolated-manifests
python scripts/run_manifest_task.py --kind stage1 \
  --manifest /path/to/new/subject-isolated-manifests/stage1_subject_isolated_15.jsonl \
  --index 0 --data-root /path/to/authorized/T-MSPD --split-root /path/to/generated/splits
```

Its grids are Stage 1 `15`, Stage 2 `30`, and Stage 3 `1800`; the six outer
target subjects never enter upstream train/validation data. Subject-dependent
clean-main checkpoints are not compatible substitutes. The historical
adaptation definition uses Stage-3 encoder/head learning rates `1e-6/1e-5`,
whereas clean-main uses Config E (`3e-6/1e-4`).

Peripheral channel removal, SmoothGrad-Abs, and the grayscale-safe Fig. 2 are
exposed by:

```text
scripts/run_channel_removal.py
scripts/run_smoothgrad.py
scripts/make_fig2_v7.py
scripts/paired_subject_statistics.py
scripts/generate_secondary_manifests.py
```

`run_subject_transfer.py` and `run_adaptation.py` are retained compatibility
wrappers for the earlier single-seed interface; they are not the strict
three-seed subject-isolated entry point.

All training analyses use validation-only selection. SmoothGrad-Abs uses 16
noise samples, noise SD `0.1 * trial EEG SD`, true-class logit gradients, and
trial-level L1 normalization. It measures model attribution, not neural source
localization.

## Expected outputs and gates

Recommended run roots keep phases isolated:

```text
runs/
  stage1/<seed>/<fold>/best_stage1.pth
  stage2/<variant>/<seed>/<fold>/best_stage2.pth
  stage3/<variant>/<seed>/<fold>/<subject>/<paradigm>/result.json
  eegnet/<seed>/<fold>/<subject>/<paradigm>/result.json
```

Before advancing phases, require complete identity grids and explicit local
success markers such as `STAGE1_SUCCESS.json`, `STAGE2_AUDIT_SUCCESS.json`,
`STAGE3_AUDIT_SUCCESS.json`, and `FINAL_SUCCESS.json`. These markers are run
artifacts and are deliberately not included in the repository.

## Audited aggregate results

The clean `outer_seed_20260811` primary analysis contains 3 seeds, 5 folds, 30
subjects, and 2 paradigms. Stage-3 balanced accuracy is averaged over 15
seed-fold cells within each subject before the 30 subjects are summarized.

| Condition | Overt BA (%) | Silent BA (%) | Stage-2 validation BA (%) |
|---|---:|---:|---:|
| Full | 58.23 | 45.32 | 48.61 |
| NoDistill | 49.15 | 38.67 | 41.23 |
| Unrestricted Shuffled-EMG | 49.29 | 38.59 | 41.23 |
| Within-class Shuffled-EMG | 57.09 | 44.72 | 48.11 |
| InstanceOnlyAlpha0 | 58.32 | 45.43 | 48.74 |
| LabelSimplex | 47.00 | 36.94 | 38.40 |

The corrected Stage-2 pooled-validation Full-minus-NoDistill difference is
`+7.38 pp` (`48.61 - 41.23`). Stage-2 values are selection validation metrics;
Stage-3 values are held-out final-test metrics.

Full minus NoDistill Stage-3 differences are `+9.08 pp` Overt (95% CI
`[8.18, 9.98]`) and `+6.65 pp` Silent (95% CI `[5.65, 7.59]`); both
two-sided sign-flip p values are below `0.001` and all `30/30` subject
differences are positive in each paradigm. Within-class shuffling retains
`87.5%/90.9%` of the unrounded Full benefit.

Full minus InstanceOnlyAlpha0 is `-0.09 pp` Overt and `-0.11 pp` Silent; both
paired bootstrap intervals include zero. Under this experiment, the
class-prototype component shows no measurable incremental value.
This is a failure to detect a difference, not evidence of equivalence.

The completed EEG-only LabelSimplex control obtained Overt BA
`47.00317084150417%` and Silent BA `36.937455199121864%`. Full minus
LabelSimplex was `+11.222402412402413 pp` Overt (95% paired-bootstrap CI
`[10.296485084360084, 12.185968061259734]`) and `+8.384680874680875 pp`
Silent (95% CI `[7.1356404690988, 9.592694732736403]`). The prespecified
two-sided sign-flip p value was below `0.001` in both paradigms, and Full was
higher for all `30/30` subjects in each. The matched fixed label-simplex
control did not reproduce the EMG-teacher benefit, showing that fixed label
geometry alone is insufficient. This does not rule out every possible
label-only regularizer.

The completed three-seed subject-isolated experiment produced zero-shot
Full/NoDistill BA of `23.63/23.52%` Overt and `20.54/20.51%` Silent. After
target-subject adaptation the corresponding values were `43.33/35.98%` and
`33.09/27.62%`. Full-minus-NoDistill was `+7.35 pp` Overt (95% CI
`[6.29, 8.49]`, 30/30 positive) and `+5.47 pp` Silent (95% CI
`[4.37, 6.63]`, 29/30 positive); both two-sided sign-flip p values were below
`0.001`.

The submitted EEGNet means are `53.26%` Overt and `40.12%` Silent, giving
Full-minus-EEGNet differences of `+4.97/+5.20 pp`, paired-bootstrap CIs
`[3.19, 6.62]/[3.40, 6.98]`, and two-sided sign-flip `p < 0.001` for both.
These submitted values are preserved unchanged. The exact author-final 60-row
subject source and calculation script were not found in the audited trees, so
their status is `EXACT_SOURCE_ARCHIVAL_PENDING`; the older complete EEGNet run
(`54.53/41.17`) is not substituted or relabeled.

In the seed-42 channel-removal analysis, the remaining-28-channel
Full-minus-NoDistill benefits are `+6.90/+4.60 pp`; attenuation relative to
the matched all-60-channel benefit is `-2.13/-2.44 pp`. SmoothGrad-Abs shows
`+1.36/+1.33 pp` of relative outer-region attribution mass, with `30/30`
positive subjects in each paradigm. These are sensitivity and relative
attribution results, not localization evidence. EMG-only and EEG+EMG observed
means are `81.71/73.11%` and `84.84/76.27%`; both require EMG at inference,
and no unarchived paired significance is claimed.

## Limitations

- Results come from one dataset.
- Silent denotes non-phonated articulation, not imagined speech.
- Benefits are sensitive to peripheral-channel removal.
- SmoothGrad-Abs is not source localization.
- The prototype term did not show measurable incremental value over paired
  instance-only distillation; this is not an equivalence conclusion.
- The within-class shuffle preserves the joint subject, paradigm, and lexical
  grouping and does not isolate a class-only causal effect.
- EEGNet is a system-level comparator, not a claim of state of the art.

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and
[docs/EXPECTED_RESULTS.md](docs/EXPECTED_RESULTS.md) for provenance and result
contracts. The submission baseline and fuller interpretation are in
[docs/SUBMITTED_RESULTS_20260919.json](docs/SUBMITTED_RESULTS_20260919.json)
and [docs/RESULTS_AND_INTERPRETATION.md](docs/RESULTS_AND_INTERPRETATION.md).
Known source and archival limitations are recorded in
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Citation and license

See [CITATION.cff](CITATION.cff). The source is MIT licensed. This repository
provides reproducibility materials for a submitted manuscript and does not
state or imply acceptance.
