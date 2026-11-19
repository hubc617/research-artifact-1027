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
and paradigm. For the grid, it averages folds within seed and then averages
seeds before forming the 30 paired participant differences. The output
reports percentage-point differences, separate random seeds for the 50,000
bootstrap and 200,000 sign-flip draws, and the plus-one correction used for
the sign-flip test.

### Standalone target adaptation

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
The `all` preset runs the matched all-channel condition. See the
[channel mask](ARCHITECTURE_AND_PROTOCOL.md#channel-mask) for the experiment scope.

Generate the SmoothGrad manifest with the paths for your task outputs.

```bash
python scripts/generate_secondary_manifests.py \
  --checkpoint-root /path/to/checkpoints \
  --run-root /path/to/runs \
  --output-dir /path/to/secondary-manifests
```

For each of the 600 tasks in `smoothgrad_seed42_600.jsonl`, supply
`channel_attribution.csv` in the directory specified by that row's `output`.
The CSV columns are `paradigm`, `variant`, `subject`, `fold`, `seed`, `channel`,
and `relative_attribution`. Repeat the task identity on every row. Include all
60 channel names. The aggregator matches channels by name in any row order.
`relative_attribution` is the channel's share of the task's mean
attribution as a fraction. For a fold-task mean attribution map, sum over the
time axis to get channel shares. The sum of the 60 shares must be within
`2e-5` of one.

The Fig. 2 computation uses 16 Gaussian draws per trial with a noise standard
deviation of 0.1 times the population standard deviation across all 60×256
trial values. Each task uses its own RNG seed (`100042` plus its zero-based
index in the final task manifest). It averages the absolute true-class logit
gradients across noise draws, L1-normalizes the trial's channel-time map,
then averages trial maps within the test fold.
`scripts/run_smoothgrad.py` is a separate analysis command. It scales noise
by each channel's sample standard deviation over time and uses the process's
RNG state. Use task CSVs produced with the Fig. 2 computation for reproduction.

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
