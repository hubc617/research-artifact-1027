# Data preparation

T-MSPD is restricted access. Obtain it from the
[official dataset page](https://cstr.cn/31253.11.sciencedb.24416) and comply
with its terms. Store your authorized copy outside the repository.

Prepare the following local `.npy` layout before generating splits.

```text
/path/to/authorized/T-MSPD/
  overt/S01/S01_eeg_feat.npy
  overt/S01/S01_emg.npy
  overt/S01/S01_audio_feat.npy
  overt/S01/S01_labels.npy
  silent_1001/S01/S01_eeg_feat.npy
  silent_1001/S01/S01_emg.npy
  silent_1001/S01/S01_labels.npy
```

EEG arrays are float32 `(N, 60, 256)`. EMG arrays have shape `(N, 6, >=1000)`. The loader uses the first 1,000 samples. Source labels are integers 1-10 and are
converted to 0-9 internally.

The split generator reads these prepared arrays and writes deterministic
subject-dependent, stratified five-fold partitions using `outer_seed_20260811`.
For a single Stage 3 task, see the [EEG-only example](../README.md#reproduction-order).

## Overt acoustic features

For Overt trials, extract features directly from the raw audio without
additional offline audio preprocessing, using the ninth Transformer block of
HuBERT Base. Save one float32 array of shape `(N, 50, 768)` per participant
under `overt/<subject>/<subject>_audio_feat.npy`. Each of the
`N` entries contains 50 time frames of 768 features. Keep the trial at each
index aligned with the EEG, EMG, and label entries at that index. The loader
checks the cache shape, dtype, finite values, and matching trial counts.

Stage 1 averages the 50 frames for each trial before its `768–512–256`
projection. Silent trials use EEG, EMG, and labels. The repository provides
neither the extraction script nor the feature cache.
