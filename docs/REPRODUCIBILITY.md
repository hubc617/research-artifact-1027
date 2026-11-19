# Reproducibility

Start with [the README commands](../README.md#reproduction-order) and the
[prepared data layout](DATA.md), including the
[Overt feature cache](DATA.md#overt-acoustic-features). The
[experiment matrix](EXPERIMENT_MATRIX.md) lists task counts and checkpoint
dependencies. Generate splits, manifests, checkpoints, and task outputs
locally. The [submitted results record](SUBMITTED_RESULTS_20260919.json)
lists the manuscript values.

## Checkpoint selection

Training selects checkpoints by validation balanced accuracy. Stage 3
reloads the selected checkpoint before constructing the final-test loader
and evaluates that partition once. See the
[training outputs](../README.md#training-outputs) for the files to check
before running a dependent task or aggregating results.

## LabelSimplex

The fixed target comes from `I_10 - 11^T/10`. The rows are L2-normalized and
placed in the first 10 coordinates of `R^256`, with zeros elsewhere.
The generated `.npy` has SHA-256
`1f0698d642af6bf2a219a0d704adbce239e9bca5c12e89f37484581ad68bf3c1`.

## EEGNet statistics

The participant-level records behind the reported EEGNet statistics are
not included in the repository.

## Paired participant statistics

`scripts/paired_subject_statistics.py` accepts a table of participant
estimates or a complete grid of five folds and three seeds per participant
and paradigm. The CSV columns are `subject`, `paradigm`, `variant`, and
`balanced_accuracy` in percent. Task-level inputs add both `seed` and `fold`.
The `--left` and `--right` conditions must each cover S01–S30 in both
`overt` and `silent_1001`. The script averages folds within each seed,
then averages the three seeds before forming the 30 paired participant differences. The output
reports percentage-point differences, separate random seeds for the 50,000
bootstrap and 200,000 sign-flip draws, and the plus-one correction used for
the sign-flip test.

## Standalone target adaptation

`scripts/run_adaptation.py` adapts an existing Stage 2 checkpoint to one
participant and paradigm. It builds its own five stratified folds within the
target (seed `20260801`) and reserves 15% of the remaining examples for
validation. It uses the common loader, so prepare the multimodal arrays in the
[data layout](DATA.md). Its output includes the train, validation, and test
counts and final-test metrics.

For the paper's subject-isolated adaptation, use
`scripts/generate_subject_isolated_manifests.py` and run the exported
`stage3_subject_isolated_1800.jsonl` tasks through
`scripts/run_manifest_task.py --kind stage3`. These tasks use the exported
train, validation, and final-test indices from `outer_seed_20260811` and
Stage 2 checkpoints trained with the target participants held out.

## Channel removal, SmoothGrad, and Fig. 2

Use `scripts/generate_secondary_manifests.py` to list the seed-42 tasks.
For each channel-removal task, `scripts/run_channel_removal.py` accepts
`--preset 28-channel` and writes metrics to the JSON path given by `--output`.
The `all` preset runs the matched all-channel condition. Both presets use EEG,
labels, the generated splits, and the matching Stage 2 checkpoint. Keep the
checkpoint fixed when comparing presets. See the
[channel mask](ARCHITECTURE_AND_PROTOCOL.md#channel-mask) for the experiment scope.

```bash
python scripts/run_channel_removal.py \
  --data-root /path/to/eeg-and-labels --split-root /path/to/generated/splits \
  --stage2-checkpoint /path/to/runs/stage2/full/seed42/fold0/best_stage2.pth \
  --subject S01 --paradigm overt --fold 0 --seed 42 --preset 28-channel \
  --output /path/to/new/28-channel.json
```

Use the seed-42 Stage 2 checkpoints for each variant and fold. Use the
validation-selected all-channel Stage 3 checkpoint with the same variant,
seed, fold, participant, and paradigm. The default Stage 3 paths follow the
`stage3/` layout under `--run-root`. The EEG and labels must follow the
[data layout](DATA.md). Point `--split-root` at the matching
`outer_seed_20260811` split files. Each task reads only its `test_idx` from
the split and uses that order.

```bash
python scripts/generate_secondary_manifests.py \
  --checkpoint-root /path/to/runs \
  --run-root /path/to/runs \
  --output-dir /path/to/secondary-manifests

python scripts/run_fig2_smoothgrad_task.py \
  --manifest /path/to/secondary-manifests/smoothgrad_seed42_600.jsonl \
  --index 0 --data-root /path/to/eeg-and-labels \
  --split-root /path/to/generated/splits
```

Run zero-based indices 0 through 599 once each before aggregating. Index 0
selects Full, fold 0, S01, Overt. Each task writes `channel_attribution.csv` to
a new directory at its manifest `output` path. To use another layout, supply
`--stage2-checkpoint` and `--stage3-checkpoint` explicitly or edit the path
fields. `--output-dir` overrides the task destination. When aggregating,
use a manifest whose `output` fields match the CSV locations.

The CSV contains `paradigm`, `variant`, `subject`, `fold`, `seed`,
`channel`, and `relative_attribution`. The 60 nonnegative channel shares are
fractions that sum to one within `2e-5`.

The Fig. 2 command uses 16 Gaussian draws per trial with a noise standard
deviation of 0.1 times the population standard deviation across all 60×256
input values. It adds noise before the EEG classifier's internal per-channel
standardization. It preserves the archived RNG consumption by first replaying
evaluation in batches of 64 with gradients disabled, then differentiating
the true-class logit for one trial's perturbations at a time. It averages
absolute gradients across noise draws, L1-normalizes the full trial map,
averages trial maps within the fold, and sums each channel over time.

The generator records the explicit `noise_seed` for each task identity
(100042 plus the zero-based index in the historical task order). Regenerate
older manifests for the new runner. Existing task CSVs can still be aggregated
using old manifests. CUDA is the default and matches the historical RNG
backend. `--device cpu` is available for smoke testing and uses the CPU RNG.
`scripts/run_smoothgrad.py` remains the general analysis command.

Aggregate the task CSVs into 120 channel differences in fraction units and two
panel annotations. The aggregator matches Full and NoDistill tasks by
participant, paradigm, and fold. It averages the five folds within each
participant and gives each participant equal weight. The panel shift is the
mean difference in the sum of the 32 peripheral channels in percentage
points. The count uses participants with positive differences.

```bash
python scripts/aggregate_smoothgrad.py \
  --manifest /path/to/secondary-manifests/smoothgrad_seed42_600.jsonl \
  --output-dir /path/to/fig2-summaries

python scripts/make_fig2_v7.py \
  --channel-summary /path/to/fig2-summaries/fig2_channel_summary.csv \
  --annotation-summary /path/to/fig2-summaries/fig2_annotation_summary.csv \
  --output /path/to/fig2-summaries/fig2.pdf
```

Install visualization and analysis packages with
`python -m pip install -e '.[visualization,analysis]'`. For a grayscale
preview, use a `.png` path for `--output` and supply a second PNG path with
`--grayscale-output`.
