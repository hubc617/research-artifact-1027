# Architecture and protocol specification

Integration specification reconciled against the executable repository where
retained evidence permits. Submission-only fields and unresolved archival
provenance are identified explicitly rather than presented as code-level
reproduction guarantees.

## Input contract

- EEG: 60 channels × 256 samples. The submitted-results record states
  preprocessing from 1,000 Hz to 256 Hz, stimulus-locked -100 to 900 ms, and
  exclusion of M1/M2/CB1/CB2. The retained frozen loader proves the 60x256
  cache contract, but the audited server trees do not retain the generation
  source needed to independently prove sampling rate, event anchor, endpoints,
  or resampling; those details remain submission-record provenance.
- EMG: six channels × 1,000 samples at 1,000 Hz, additional 150/250-Hz notch filtering.
- Ten class labels; 30 participants; 14,869 Overt and 14,851 Silent trials.
- Per-channel temporal normalization for reconstruction targets.
- HuBERT Base block 9, temporal mean pooling, 768-D trial representation. Archive the actual extraction script, checkpoint identifier/revision, sampling/resampling path, and cache identity. Do not infer historical resampling from model requirements.
- Preserve any existing cache interface until actual compatibility is checked. A shared library class whose name contains Wav2Vec2 is not evidence that the encoder was wav2vec 2.0.

## Encoder and heads

EEG/EMG patch lengths 16/25; residual depthwise temporal convolution kernel 7 and 4× pointwise expansion; Transformer 4 layers, 8 heads, dimension 256, FFN 1,024; learned CLS token. Reconstruction decoder: 2 layers, 4 heads; EEG adds convolutional smoothing.

Stage-1 acoustic projection: 768–512–256. Stage-2 EEG projection: 256–256–256 with GELU. Stage-2 auxiliary classifier: LayerNorm/dropout(0.3)/linear. Stage-3 new classifier: 256–256–10, with the documented LayerNorm/BatchNorm/GELU/dropout(0.5) mechanics; verify the exact module order from the frozen runner before copying code.

## Objectives and data access

Stage 1: masked MSE + 0.1 CE + symmetric InfoNCE (temperature 0.07); acoustic alignment on Overt only. Mask ratio 0.75. Freeze the trained EMG encoder afterward. EMG weights do not initialize trainable EEG parameters.

Stage 2: 30 reconstruction-only warm-up epochs; afterward reconstruction + 0.1 CE + distillation for Full. EEG masking 0.75. Unit-normalize projected EEG and teacher embeddings. Full teacher target is the normalized sum of the paired EMG unit vector and α=0.3 times the mean of other same-class mini-batch EMG unit vectors. Empty class neighborhoods fall back to the paired vector. Mean cosine distance supplies distillation. Never route EEG into teacher-target construction.

NoDistill removes distillation only. Instance-only sets α=0 only. Within-class is a fixed train-only, one-to-one, zero-fixed-point derangement within subject×paradigm×class. Unrestricted uses a seeded nonzero cyclic shift per mini-batch. These are different algorithms, not aliases.

LabelSimplex removes teacher construction, EMG checkpoint loading, and EMG trial reads. Use the frozen ten-class 256-D regular simplex and label indexing. Verify the frozen tensor hash against the canonical source rather than generating an arbitrary alternative basis with the same shape. Same-seed trainable EEG initialization must match the reference pipeline.

Stage 3: replace the auxiliary classifier and jointly fine-tune encoder/new head on EEG with CE; no input masking and no EMG access. For each task select the checkpoint on validation, reload it, then construct and evaluate the final-test loader. Operational retries must never choose among successful results by their test performance.

## Optimization

| Setting | Stage 1 | Stage 2 | Stage 3 |
| --- | --- | --- | --- |
| Epochs | 150 | 150 | 100 |
| Batch size | 64 | 8 | 8 |
| Initial LR | 5e-4 | 3e-4 | encoder 3e-6; head 1e-4 |
| Cosine LR floor | 1e-6 | 1e-6 | 1e-6 |
| Mask ratio | 0.75 | 0.75 | none |

AdamW weight decay 0.05; CE label smoothing 0.1; gradient norm clip 1.0. Stage-3 settings were selected on validation and frozen across variants. The clean eight-candidate manifest is the authority for the search space; do not reuse historical pilot validation scores as clean results.

## Channel mask

Remaining 28: F3, F1, FZ, F2, F4, FC3, FC1, FCZ, FC2, FC4, C3, C1, CZ, C2, C4, CP3, CP1, CPZ, CP2, CP4, P3, P1, PZ, P2, P4, PO3, POZ, PO4.

Removed 32: FP1, FPZ, FP2, AF3, AF4, F7, F5, F6, F8, FT7, FC5, FC6, FT8, T7, C5, C6, T8, TP7, CP5, CP6, TP8, P7, P5, P6, P8, PO7, PO5, PO6, PO8, O1, OZ, O2.

Resolve names against the actual montage, validate disjointness and 60-channel coverage, and hash the ordered mask. Keep the tensor shape; mask multiplicatively in Stage-3 train/validation/test and retrain. Stage-2 checkpoints remain all-channel.

## SmoothGrad and Fig. 2

Fixed seed-42 Full/NoDistill Stage-3 models; true-class logit; 16 noisy samples per trial; noise SD=0.1×trial EEG SD. Use the archived SmoothGrad-Abs implementation, with trial-level L1 normalization before participant aggregation. Preserve source-defined aggregation and absolute-gradient ordering; do not silently switch between absolute mean gradient and mean absolute gradient.

Render with the final grayscale-safe v7 source, using fresh numerical inputs. Check annotation units, sign/contours, labels at physical insertion size, clipping, embedded TrueType fonts, and no Type 3 fonts. This source-only preparation must not commit result arrays or figure PDFs.
