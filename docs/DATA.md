# Data preparation

T-MSPD is restricted access. Obtain it from the
[official dataset page](https://cstr.cn/31253.11.sciencedb.24416) and comply
with its terms. Data and derived arrays are not distributed in this repository.

Expected local layout:

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

EEG arrays are float32 `(N, 60, 256)`. EMG arrays are `(N, 6, >=1000)`;
the first 1000 samples are used. Source labels are integers 1-10 and are
converted to 0-9 internally.

For Overt trials, `*_audio_feat.npy` is a compatibility filename for float32
HuBERT Base features from the ninth Transformer block with shape `(N, 50,
768)`. Features are temporally mean-pooled to 768 dimensions before the
`768-512-256` projection. Silent trials have no acoustic feature file. Users
must generate this cache themselves; neither audio nor cached features are
provided here.

The loader validates dtype, shape, finite values, and positional trial-count
alignment. Splits are deterministic subject-dependent stratified five-fold
partitions from `outer_seed_20260811`.
