# Reproducibility materials — Submission 1027

This repository provides code and configurations for Privileged EMG Distillation for EEG-Only Speech Decoding. EMG serves as privileged information during training. Stage 3 fine-tuning and inference use EEG only.

Under the reported protocol, Full improves balanced accuracy over NoDistill and the included EEGNet-8,2 baseline.

## Method

1. **Stage 1, EMG teacher.** A masked autoencoder learns from raw EMG, with
   lexical classification and Overt-only alignment using [cached acoustic
   features](docs/DATA.md#overt-acoustic-features). Stage 1 produces the
   frozen EMG teacher for Stage 2.
2. **Stage 2, EEG pretraining.** An EEG masked autoencoder trains for 150
   epochs. Epochs 1-30 use reconstruction only. Thereafter the configured
   representation loss and `0.1 * cross_entropy` are added. Checkpoints are
   selected by pooled validation balanced accuracy. In distillation
   conditions, EMG generates the target while EEG enters the student loss.
3. **Stage 3, EEG-only fine-tuning.** A subject-dependent classifier is trained
   for 100 epochs on the clean `outer_seed_20260811` five-fold split. See the
   [validation and final-test order](docs/REPRODUCIBILITY.md#checkpoint-selection).

The six Stage 2 conditions use these targets and controls.

- `full` adds the same-class prototype term with `alpha=0.3` to the paired EMG target.
- `no_distill` uses reconstruction and auxiliary classification.
- `unrestricted_shuffle` rotates the EMG donors within each training mini-batch.
- `within_class_shuffle` reassigns each training example to a different EMG
  donor in the same subject, paradigm, and lexical class.
- `instance_only_alpha0` uses the paired EMG target with `alpha=0`.
- `label_simplex` uses EEG and lexical labels. Each label selects a row of a
  fixed 10-class regular simplex embedded in 256 dimensions. The target loss
  has weight `1.0` after the same warm-up.

## Data and acoustic features

See [data preparation](docs/DATA.md) for T-MSPD access, the expected local
arrays, and the [Overt feature cache](docs/DATA.md#overt-acoustic-features).
Supply local paths through the CLI or a copy of `configs/paths.example.yaml`.

The [implementation crosswalk](docs/IMPLEMENTATION_CROSSWALK.md) links
the experimental protocol to the repository code.

## Environment

Python 3.10+ and a CUDA-compatible PyTorch installation are required for
training. Tested dependency versions are in
[TESTED_ENVIRONMENT.md](TESTED_ENVIRONMENT.md).

```bash
pip install -e '.[analysis,visualization,test]'
```

## Reproduction order

Prepare the local `.npy` directory tree described in [data preparation](docs/DATA.md)
before running `prepare_splits.py`. This command uses the common multimodal loader
and needs EEG, EMG, labels, and Overt acoustic features to generate the splits.

```bash
python scripts/prepare_splits.py \
  --data-root /path/to/authorized/T-MSPD \
  --output /path/to/generated/splits
```

Generate task manifests in a new directory.

```bash
python scripts/generate_protocol_manifests.py \
  --data-root /path/to/authorized/T-MSPD \
  --split-root /path/to/generated/splits \
  --run-root /path/to/runs \
  --output-dir /path/to/manifests
```

The generator writes `stage1_15.jsonl` for the 15 shared Stage 1 tasks.
The six-condition grid uses `stage2_six_variant_90.jsonl` and
`stage3_six_variant_5400.jsonl`. The four-condition grid uses
`stage2_clean_main_four_variant_60.jsonl` and
`stage3_clean_main_four_variant_3600.jsonl`.

Run Stage 1, then Stage 2 for the same seed and fold. The full grid uses seeds
`42`, `2026`, and `3407` and folds `0–4`. These examples use seed 42 and fold 0.

```bash
python scripts/train_stage1.py --config configs/stage1.yaml \
  --data-root /path/to/authorized/T-MSPD --split-root /path/to/generated/splits \
  --seed 42 --fold 0 --epochs 150 --output /path/to/runs/stage1/seed42/fold0

python scripts/train_stage2.py --config configs/stage2_full.yaml \
  --data-root /path/to/authorized/T-MSPD --split-root /path/to/generated/splits \
  --seed 42 --fold 0 --epochs 150 --variant full \
  --stage1-checkpoint /path/to/runs/stage1/seed42/fold0/best_stage1.pth \
  --output /path/to/runs/stage2/full/seed42/fold0

python scripts/train_stage2.py --config configs/stage2_instance_only_alpha0.yaml \
  --data-root /path/to/authorized/T-MSPD --split-root /path/to/generated/splits \
  --seed 42 --fold 0 --epochs 150 --variant instance_only_alpha0 \
  --stage1-checkpoint /path/to/runs/stage1/seed42/fold0/best_stage1.pth \
  --output /path/to/runs/stage2/instance_only_alpha0/seed42/fold0

python scripts/train_stage2.py --config configs/stage2_label_simplex.yaml \
  --data-root /path/to/authorized/T-MSPD --split-root /path/to/generated/splits \
  --seed 42 --fold 0 --epochs 150 --variant label_simplex \
  --output /path/to/runs/stage2/label_simplex/seed42/fold0
```

Stage 3 fine-tunes on EEG and labels using an existing split and the Stage 2
checkpoint for the matching variant, seed, and fold. Obtain that checkpoint
by running Stage 2. The example runs one Overt task. Its EEG and label
files follow the same subject and paradigm
layout described in [data preparation](docs/DATA.md).

```bash
python scripts/train_stage3.py --config configs/stage3.yaml \
  --data-root /path/to/eeg-and-labels --split-root /path/to/generated/splits \
  --seed 42 --fold 0 --epochs 100 --subject S01 --paradigm overt --variant full \
  --stage2-checkpoint /path/to/runs/stage2/full/seed42/fold0/best_stage2.pth \
  --output /path/to/runs/stage3/full/seed42/fold0/S01/overt
```

For the full Stage 3 grid, run all 30 subjects, both paradigms, five folds,
and three seeds.

For LabelSimplex Stage 3, use `configs/stage3_label_simplex.yaml`, pass
`--variant label_simplex`, and provide the corresponding validation-selected
LabelSimplex Stage 2 checkpoint.

Use a fresh output directory for each training task.

## EEGNet and secondary analyses

EEGNet-8,2 trains from scratch using EEG and labels with the prepared splits.
Generate the manifest and run one task with `--index 0`.

```bash
python scripts/generate_eegnet_manifest.py \
  --data-root /path/to/authorized/T-MSPD --split-root /path/to/generated/splits \
  --run-root /path/to/runs/eegnet --output-dir /path/to/manifests/eegnet
python scripts/train_eegnet.py \
  --manifest /path/to/manifests/eegnet/eegnet_3seed_900tasks.jsonl --index 0
```

Run each manifest index from `0` through `899` before aggregating. The
aggregator checks the files for each task and writes `eegnet_final_summary.csv`
alongside the task-level and participant-level summaries.

```bash
python scripts/aggregate_eegnet.py \
  --manifest /path/to/manifests/eegnet/eegnet_3seed_900tasks.jsonl \
  --out-dir /path/to/summaries/eegnet
```

For the three-seed subject-isolated evaluation, generate the upstream task
grid and export separate train, validation, and test index files for the targets. This example runs one
of the upstream Stage 1 tasks.

```bash
python scripts/generate_subject_isolated_manifests.py \
  --source-split-root /path/to/generated/splits \
  --run-root /path/to/runs/subject-isolated \
  --output-dir /path/to/manifests/subject-isolated
python scripts/run_manifest_task.py --kind stage1 \
  --manifest /path/to/manifests/subject-isolated/stage1_subject_isolated_15.jsonl \
  --index 0 --data-root /path/to/authorized/T-MSPD --split-root /path/to/generated/splits
```

The [experiment matrix](docs/EXPERIMENT_MATRIX.md#subject-isolated-transfer-and-adaptation)
describes the target exclusions, task counts, and checkpoint lineage.

See the [secondary analysis instructions](docs/REPRODUCIBILITY.md#channel-removal-smoothgrad-and-fig-2)
for channel removal, SmoothGrad-Abs, and Fig. 2.

## Training outputs

Each Stage 1, 2, or 3 task writes `result.json` and its validation-selected
checkpoint. The checkpoints for the example tasks are at these paths.

```text
runs/
  stage1/seed42/fold0/best_stage1.pth
  stage2/full/seed42/fold0/best_stage2.pth
  stage3/full/seed42/fold0/S01/overt/best_stage3.pth
  eegnet/seed42/fold0/S01/overt/best_eegnet.pth
```

Before starting the next stage, check that the prerequisite tasks have both
their result and checkpoint files. For a full-grid summary, check every row
in the selected manifest. EEGNet aggregation requires `result.json` with
`"status": "complete"`, `best_eegnet.pth`, and `predictions.npz` for every task.
The aggregator checks that the predictions match the test splits.

Summarize completed main-grid tasks from their manifests.

```bash
python scripts/aggregate_paper_results.py --grid four \
  --stage2-manifest /path/to/manifests/stage2_clean_main_four_variant_60.jsonl \
  --stage3-manifest /path/to/manifests/stage3_clean_main_four_variant_3600.jsonl \
  --output /path/to/new/main-grid-summary
```

Use `--grid six` with `stage2_six_variant_90.jsonl` and
`stage3_six_variant_5400.jsonl` to include InstanceOnly and LabelSimplex.
For separate batches, pass each manifest of a stage with another
`--stage2-manifest` or `--stage3-manifest` flag. Use absolute paths in the
manifests, or run the aggregator from the working directory used for the tasks.
Every selected task must have an identity-matched `out_dir/result.json`.

The summary uses balanced accuracy in percent. Stage 3 first averages each
participant's five folds and three seeds, then averages across participants
within each condition and paradigm. Paired differences for the five core
contrasts are in percentage points. The optional Stage 2 table averages
validation balanced accuracy over the 15 seed/fold cells for each condition.

## Reported results

The clean `outer_seed_20260811` primary analysis contains 3 seeds, 5 folds, 30
subjects, and 2 paradigms. Stage-3 balanced accuracy is averaged over 15
seed-fold cells within each subject before the 30 subjects are summarized.
Silent denotes non-phonated articulation.

| Condition | Overt BA (%) | Silent BA (%) | Stage-2 validation BA (%) |
|---|---:|---:|---:|
| Full | 58.23 | 45.32 | 48.61 |
| NoDistill | 49.15 | 38.67 | 41.23 |
| Unrestricted Shuffled-EMG | 49.29 | 38.59 | 41.23 |
| Within-class Shuffled-EMG | 57.09 | 44.72 | 48.11 |
| InstanceOnlyAlpha0 | 58.32 | 45.43 | 48.74 |
| LabelSimplex | 47.00 | 36.94 | 38.40 |

The Stage 2 pooled-validation Full-minus-NoDistill difference is
`+7.38 pp` (`48.61 - 41.23`). Stage 2 values come from checkpoint selection
on validation data. Stage 3 values come from the held-out final test.

Full minus NoDistill Stage-3 differences are `+9.08 pp` Overt (95% CI
`[8.18, 9.98]`) and `+6.65 pp` Silent (95% CI `[5.65, 7.59]`); both
two-sided sign-flip p values are below `0.001` and all `30/30` subject
differences are positive in each paradigm. Within-class shuffling retains
`87.5%/90.9%` of the unrounded Full benefit.
The shuffle preserves the joint subject, paradigm, and lexical grouping.

Full minus InstanceOnlyAlpha0 is `-0.09 pp` Overt and `-0.11 pp` Silent; both
paired bootstrap intervals include zero.

The completed EEG-only LabelSimplex control obtained Overt BA
`47.00317084150417%` and Silent BA `36.937455199121864%`. Full minus
LabelSimplex was `+11.222402412402413 pp` Overt (95% paired-bootstrap CI
`[10.296485084360084, 12.185968061259734]`) and `+8.384680874680875 pp`
Silent (95% CI `[7.1356404690988, 9.592694732736403]`). The prespecified
two-sided sign-flip p value was below `0.001` in both paradigms, and Full was
higher for all `30/30` subjects in each paradigm.

With six target participants excluded per upstream fold, the three-seed
subject-isolated experiment produced zero-shot Full/NoDistill BA of `23.63/23.52%` Overt and `20.54/20.51%` Silent. After
supervised EEG-only fine-tuning on the targets, the corresponding values
were `43.33/35.98%` and `33.09/27.62%`. Full-minus-NoDistill was `+7.35 pp` Overt (95% CI
`[6.29, 8.49]`, 30/30 positive) and `+5.47 pp` Silent (95% CI
`[4.37, 6.63]`, 29/30 positive); both two-sided sign-flip p values were below
`0.001`.

The submitted EEGNet means are `53.26%` Overt and `40.12%` Silent, giving
Full-minus-EEGNet differences of `+4.97/+5.20 pp`, paired-bootstrap CIs
`[3.19, 6.62]/[3.40, 6.98]`, and two-sided sign-flip `p < 0.001` for both.
See the [EEGNet reproduction note](docs/REPRODUCIBILITY.md#eegnet-statistics).

In the seed-42 channel-removal analysis, the remaining-28-channel
Full-minus-NoDistill benefits are `+6.90/+4.60 pp`. The differences from the
matched all-60-channel benefits are `-2.13/-2.44 pp`. SmoothGrad-Abs shows
`+1.36/+1.33 pp` of relative outer-region model attribution mass, with `30/30`
positive subjects in each paradigm. EMG-only and EEG+EMG observed means are
`81.71/73.11%` and `84.84/76.27%`. Both use EMG at inference.

See the [submitted results record](docs/SUBMITTED_RESULTS_20260919.json),
[full-precision results](docs/EXPECTED_RESULTS.md), and
[interpretation](docs/RESULTS_AND_INTERPRETATION.md).

## Citation and license

See [CITATION.cff](CITATION.cff). The source is MIT licensed. This repository
provides reproducibility materials for a submitted manuscript.
