# Experimental matrix and checkpoint lineage

## Subject-dependent main evaluation

Seeds: 42, 2026, 3407. Five stratified trial-level folds within participant and paradigm, approximately 68/12/20 percent train/validation/test. Upstream pooling uses only allowed training trials; validation selects upstream checkpoints. Final-test trials are excluded from upstream learning and model selection.

| Component | Grid | Count |
| --- | --- | ---: |
| Shared Stage 1 | 3 seeds × 5 folds | 15 |
| Four original Stage-2 variants | 4 × 3 × 5 | 60 |
| Four-variant Stage 3 | 4 × 3 × 5 × 30 subjects × 2 paradigms | 3,600 |
| Instance-only Stage 2 / Stage 3 | one additional variant | 15 / 900 |
| LabelSimplex Stage 2 / Stage 3 | one additional variant; no EMG teacher | 15 / 900 |
| Six-condition total Stage 2 / Stage 3 | excluding duplicate shared upstream work | 90 / 5,400 |
| EEGNet intended grid | 3 × 5 × 30 × 2 | 900 |

EEGNet's selected run must have its exact task/subject provenance archived. A manifest with 900 planned rows is not itself proof that 900 results were produced or aggregated.

## Subject-isolated transfer and adaptation

Five outer folds hold out six participants each. Upstream training and validation exclude all targets. Three seeds are used.

| Component | Grid | Count |
| --- | --- | ---: |
| Separate EMG teachers | 3 seeds × 5 outer folds | 15 |
| Separate Full/NoDistill upstream EEG | 2 × 3 × 5 outer folds | 30 |
| Target EEG Stage-3 adaptation | 2 × 3 × 30 subjects × 2 paradigms × 5 target folds | 1,800 |

Zero-shot uses upstream auxiliary heads without target updates. Its manifest and aggregation must be archived separately; do not infer zero-shot task count from the adaptation grid. The subject-dependent checkpoints cannot substitute for this target-excluded lineage.

## Secondary analyses

Channel-removal retraining: 2 variants × 30 subjects × 2 paradigms × 5 folds × seed 42 = 600 tasks. The 600 all-channel references are reused, giving 1,200 comparison rows rather than 1,200 new retraining jobs.

SmoothGrad: 600 analysis tasks on the same two-variant seed-42 identity grid. Sixteen noisy samples per trial are not sixteen independent scientific observations.

EMG-only and EEG+EMG contextual results require their own historical manifest/protocol archive. Do not invent task counts or paired significance from the reported means.

## Identity and validity fields

For each result retain variant, seed, participant, paradigm, fold(s), source commit/hash, ordered split identity, upstream checkpoint hash, configuration, selected epoch, validation criterion, final-test boundary, attempt identity, and completion marker. Exclude smoke tasks by manifest membership. Do not relabel historical mixed-split artifacts as clean. Failed or incomplete phases cannot become successful by manually editing a marker.
