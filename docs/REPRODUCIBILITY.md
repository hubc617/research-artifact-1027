# Reproducibility status

The source exposes explicit data, split, checkpoint, and output arguments and
contains no private server default. The clean split uses seeds 42, 2026, and
3407 with five folds. Stage 1 and Stage 2 use 150 epochs; Stage 2 starts with a
30-epoch reconstruction warm-up. Stage 3 uses the frozen Config-E setting:
encoder LR `3e-6`, classifier LR `1e-4`, dropout `0.5`, label smoothing `0.1`,
100 epochs, AdamW weight decay `0.05`, gradient clipping `1.0`, and cosine
annealing to `1e-6`.

Checkpoint selection is based on validation balanced accuracy. Primary Stage 3
constructs the final-test partition only after checkpoint selection and strict
reload, then evaluates it once. Final-test metrics must never drive selection,
ranking, retry, or stopping.

LabelSimplex uses `configs/stage2_label_simplex.yaml` and the same seeded EEG
student construction as clean Full. Its Stage-2 loader resolves only EEG and
lexical-label arrays, and a process audit hook rejects EMG trials, Stage-1
checkpoints, and EMG-encoder artifacts. The fixed target is generated from
`I_10 - 11^T/10`, row-normalized, embedded in the first 10 coordinates of
`R^256`, and zero-padded. The authoritative generated `.npy` byte hash is
`1f0698d642af6bf2a219a0d704adbce239e9bca5c12e89f37484581ad68bf3c1`.
Its compatibility manifests contain 15 Stage-2 and 900 Stage-3 tasks. The
unified six-condition manifests contain 90 Stage-2 and 5,400 Stage-3 cells;
the independent clean-main four-condition manifests remain 60 and 3,600.

The acoustic cache contract is HuBERT Base, ninth Transformer block, temporal
mean pooling, 768-D trial representation, and a `768-512-256` projection MLP.
The cache API remains `*_audio_feat.npy` for backward compatibility. The source
does not generate or distribute audio features, and AttentionPooling is not a
method component. The historical extractor, exact model revision, source audio
sampling rate, and any resampling operation are not archived here and must not
be inferred from the 768-D cache shape.

The strict subject-isolated generator emits 15/30/1,800 Stage-1/2/3 task grids
and separate target train/validation/final-test role files. Each outer fold
excludes all six target subjects from upstream training and validation. This
pipeline preserves the historical adaptation learning rates `1e-6/1e-5`; it
must not use subject-dependent clean-main checkpoints.

`paired_subject_statistics.py` accepts either one subject estimate per cell or
strict 15-cell seed/fold inputs. For raw inputs it first averages five folds
within each seed, then averages the three seeds, and finally performs inference
on 30 subject differences. Outputs record separate bootstrap and sign-flip RNG
seeds, 50,000/200,000 draws by default, percentage-point units, and the
plus-one sign-flip correction.

The repository includes only source and documentation. Generated manifests,
splits, checkpoints, task results, predictions, curves, caches, PDFs, and raw
or derived arrays are excluded. Public release checks verify the committed file
set directly; the detailed internal release audit is retained outside this
repository.
