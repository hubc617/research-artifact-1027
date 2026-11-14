# Submission-to-source crosswalk

The [submitted results record](SUBMITTED_RESULTS_20260919.json) gives the
manuscript values. The table points to the public implementation of each
experiment.

| Experiment or method | Public implementation |
|---|---|
| Overt acoustic inputs | [Feature cache format](DATA.md#overt-acoustic-features), [loader](../src/privileged_emg/dataset.py), and [Stage 1 model](../src/privileged_emg/model_stage1.py) |
| EEG epochs and input shape | [Protocol](ARCHITECTURE_AND_PROTOCOL.md#input-contract) and [loader](../src/privileged_emg/dataset.py) |
| Four-layer, eight-head, 256-D encoder with 1024-D FFN | [Shared MAE](../src/privileged_emg/model_MAE.py) |
| EEG/EMG patch sizes 16/25 | [Stage 2 model](../src/privileged_emg/model_stage2.py) |
| Full, NoDistill, and both shuffling controls | [Training chain](../src/privileged_emg/chain.py) and [configuration files](../configs) |
| Instance-only (α=0) | [Stage 2 configuration](../configs/stage2_instance_only_alpha0.yaml) |
| LabelSimplex | [EEG loader and fixed targets](../src/privileged_emg/label_simplex.py) and [Stage 2 configuration](../configs/stage2_label_simplex.yaml) |
| Six-condition task grid | [Manifest generator](../scripts/generate_protocol_manifests.py) and [experiment matrix](EXPERIMENT_MATRIX.md) |
| Config E validation selection | [Eight validation candidates](../configs/stage3_validation_search.yaml) and [Stage 3 configuration](../configs/stage3.yaml) |
| Single Stage 3 task | [EEG-only command](../README.md#reproduction-order) and [Stage 3 entry point](../scripts/train_stage3.py) |
| Three-seed subject isolation | [Manifest generator](../scripts/generate_subject_isolated_manifests.py) and [task runner](../scripts/run_manifest_task.py) |
| EEGNet-8,2 | [Manifest, training, and aggregation commands](../README.md#eegnet-and-secondary-analyses) |
| 28-channel removal, SmoothGrad-Abs, and Fig. 2 | [Secondary analysis commands](REPRODUCIBILITY.md#channel-removal-smoothgrad-and-fig-2) |
| EMG-only and EEG+EMG fusion | [Modality baseline runner](../scripts/run_modality_baselines.py) |
