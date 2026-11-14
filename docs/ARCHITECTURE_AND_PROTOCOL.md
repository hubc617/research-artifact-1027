# Architecture and protocol specification

## Input contract

- The loader reads EEG arrays with 60 channels × 256 samples. The
  [submitted results record](SUBMITTED_RESULTS_20260919.json) describes
  stimulus-locked EEG epochs from −100 to 900 ms, conversion from 1,000 Hz
  to 256 Hz, and exclusion of M1/M2/CB1/CB2.
- EMG uses six channels × 1,000 samples at 1,000 Hz with additional
  150/250-Hz notch filtering.
- The dataset contains ten classes and 30 participants, with 14,869 Overt
  and 14,851 Silent trials.
- Reconstruction targets are normalized over time per channel.
- See [Overt acoustic features](DATA.md#overt-acoustic-features) for the
  acoustic inputs to Stage 1.

## Encoder and heads

EEG/EMG patch lengths 16/25; residual depthwise temporal convolution kernel 7 and 4× pointwise expansion; Transformer 4 layers, 8 heads, dimension 256, FFN 1,024; learned CLS token. Reconstruction decoder: 2 layers, 4 heads; EEG adds convolutional smoothing.

Stage-2 EEG projection: 256–256–256 with GELU. Stage-2 auxiliary classifier: LayerNorm/dropout(0.3)/linear. Stage 3 uses a 256–256–10 classifier with LayerNorm, a linear layer, BatchNorm, GELU, dropout(0.5), and a final linear layer, in that order.

## Objectives and data access

Stage 1: masked MSE + 0.1 CE + symmetric InfoNCE (temperature 0.07); acoustic alignment on Overt only. Mask ratio 0.75. The trained EMG encoder supplies frozen teacher targets during Stage 2, which initializes its own EEG encoder.

Stage 2: 30 reconstruction-only warm-up epochs; afterward reconstruction + 0.1 CE + distillation for Full. EEG masking 0.75. Unit-normalize projected EEG and teacher embeddings. Full teacher target is the normalized sum of the paired EMG unit vector and α=0.3 times the mean of other same-class mini-batch EMG unit vectors. Empty class neighborhoods fall back to the paired vector. Mean cosine distance supplies distillation. Teacher targets are computed from EMG embeddings.

NoDistill removes distillation only. Instance-only sets α=0 only. Within-class is a fixed train-only, one-to-one, zero-fixed-point derangement within subject×paradigm×class. Unrestricted uses a seeded nonzero cyclic shift per mini-batch.

LabelSimplex uses EEG and labels, with a fixed ten-class regular simplex embedded in 256 dimensions. The EEG initialization matches Full for the same seed.

Stage 3 replaces the auxiliary classifier and jointly fine-tunes the encoder and new head on unmasked EEG using cross-entropy. See [checkpoint selection](REPRODUCIBILITY.md#checkpoint-selection) for the evaluation order.

## Optimization

| Setting | Stage 1 | Stage 2 | Stage 3 |
| --- | --- | --- | --- |
| Epochs | 150 | 150 | 100 |
| Batch size | 64 | 8 | 8 |
| Initial LR | 5e-4 | 3e-4 | encoder 3e-6; head 1e-4 |
| Cosine LR floor | 1e-6 | 1e-6 | 1e-6 |
| Mask ratio | 0.75 | 0.75 | none |

AdamW weight decay 0.05; CE label smoothing 0.1; gradient norm clip 1.0. Stage-3 settings were selected on validation and frozen across variants. The eight validation candidates are listed in [stage3_validation_search.yaml](../configs/stage3_validation_search.yaml).

## Channel mask

Remaining 28: F3, F1, FZ, F2, F4, FC3, FC1, FCZ, FC2, FC4, C3, C1, CZ, C2, C4, CP3, CP1, CPZ, CP2, CP4, P3, P1, PZ, P2, P4, PO3, POZ, PO4.

Removed 32: FP1, FPZ, FP2, AF3, AF4, F7, F5, F6, F8, FT7, FC5, FC6, FT8, T7, C5, C6, T8, TP7, CP5, CP6, TP8, P7, P5, P6, P8, PO7, PO5, PO6, PO8, O1, OZ, O2.

The seed-42 comparison covers Full and NoDistill on the same 30 participants
and five folds per paradigm. Both conditions start from all-channel Stage 2
checkpoints. The 28-channel condition preserves the 60-channel tensor and
sets the listed 32 channels to zero during Stage 3 training, validation, and
final testing.

## SmoothGrad and Fig. 2

SmoothGrad-Abs summarizes relative model attribution to each channel.
The seed-42 analysis uses Full and NoDistill Stage 3 models for all 30
participants and both paradigms. For each trial, the script takes absolute
gradients of the true-class logit for 16 noisy inputs, with noise SD equal
to 0.1 times the trial EEG SD. It averages across noise draws and time,
then L1-normalizes the channel values before aggregation.

See [secondary analysis instructions](REPRODUCIBILITY.md#channel-removal-smoothgrad-and-fig-2)
for the plotting input format.
