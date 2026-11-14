# Experimental matrix and checkpoint lineage

## Subject-dependent main evaluation

Seeds: 42, 2026, 3407. Five stratified trial-level folds within participant and paradigm, approximately 68/12/20 percent train/validation/test. Upstream fits pool training trials and select checkpoints on the validation partitions.

| Component | Grid | Count |
| --- | --- | ---: |
| Shared Stage 1 | 3 seeds × 5 folds | 15 |
| Four original Stage-2 variants | 4 × 3 × 5 | 60 |
| Four-variant Stage 3 | 4 × 3 × 5 × 30 subjects × 2 paradigms | 3,600 |
| Instance-only Stage 2 / Stage 3 | one additional variant | 15 / 900 |
| LabelSimplex Stage 2 / Stage 3 | one additional variant; no EMG teacher | 15 / 900 |
| Six-condition total Stage 2 / Stage 3 | excluding duplicate shared upstream work | 90 / 5,400 |
| EEGNet-8,2 | 3 × 5 × 30 × 2 | 900 |

See [EEGNet training and aggregation](../README.md#eegnet-and-secondary-analyses) for the command sequence.

## Subject-isolated transfer and adaptation

Five outer folds hold out six participants each. Upstream training and validation exclude all targets. Three seeds are used.

| Component | Grid | Count |
| --- | --- | ---: |
| Separate EMG teachers | 3 seeds × 5 outer folds | 15 |
| Separate Full/NoDistill upstream EEG | 2 × 3 × 5 outer folds | 30 |
| Target EEG Stage-3 adaptation | 2 × 3 × 30 subjects × 2 paradigms × 5 target folds | 1,800 |

Zero-shot evaluation uses the upstream auxiliary head on the held-out participants. Supervised adaptation fine-tunes the encoder and a new classifier on labeled EEG from each target's training partition. Adaptation uses encoder and classifier learning rates of `1e-6` and `1e-5`, respectively. Both evaluations start from checkpoints trained and selected with the six target participants excluded.

## Secondary analyses

Channel-removal retraining covers Full and NoDistill, 30 participants, two
paradigms, five folds, and seed 42. The 600 retrained tasks are paired with
600 matched all-channel results for 1,200 comparison rows.

SmoothGrad-Abs has 600 analyses on the all-channel Stage 3 checkpoints from
the same seed-42 grid. See [architecture and protocol](ARCHITECTURE_AND_PROTOCOL.md)
for the channel mask and attribution calculation.
