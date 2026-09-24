import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

EMG_RAW_LEN = 1000
EMG_FS = 1000.0
EMG_NOTCH_FREQS = (150.0, 250.0)
EMG_NOTCH_HALF_WIDTH_HZ = 4.0


def _require_finite(name, array, path):
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f'{path}: {name} must contain only finite numeric values')


def _validate_trial_alignment(eeg, emg, audio, labels, eeg_path, emg_path, audio_path, label_path):
    """Validate the retained positional trial-alignment contract before pooling."""
    if eeg.ndim != 3 or eeg.shape[1:] != (60, 256):
        raise ValueError(f'{eeg_path}: expected EEG shape (N, 60, 256), got {eeg.shape}')
    if emg.ndim != 3 or emg.shape[1] != 6 or emg.shape[-1] < EMG_RAW_LEN:
        raise ValueError(f'{emg_path}: expected EMG shape (N, 6, >= {EMG_RAW_LEN}), got {emg.shape}')
    if audio.dtype != np.float32:
        raise ValueError(f'{audio_path}: expected audio dtype float32, got {audio.dtype}')
    if audio.ndim != 3 or audio.shape[1:] != (50, 768):
        raise ValueError(f'{audio_path}: expected audio shape (N, 50, 768), got {audio.shape}')
    if labels.ndim != 1:
        raise ValueError(f'{label_path}: expected labels shape (N,), got {labels.shape}')

    for name, array, path in (
        ('EEG', eeg, eeg_path),
        ('EMG', emg, emg_path),
        ('audio', audio, audio_path),
        ('labels', labels, label_path),
    ):
        _require_finite(name, array, path)

    counts = {'EEG': len(eeg), 'EMG': len(emg), 'audio': len(audio), 'labels': len(labels)}
    if len(set(counts.values())) != 1:
        raise ValueError(
            f'positional trial alignment requires equal trial counts; got {counts}'
        )


def _notch_coeffs(freq_hz, fs, half_width_hz):
    bandwidth_hz = 2.0 * half_width_hz
    q = freq_hz / bandwidth_hz
    w0 = 2.0 * np.pi * freq_hz / fs
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)

    b0, b1, b2 = 1.0, -2.0 * cos_w0, 1.0
    a0, a1, a2 = 1.0 + alpha, -2.0 * cos_w0, 1.0 - alpha
    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return b, a


def _lfilter_biquad(x, b, a):
    x64 = x.astype(np.float64, copy=False)
    y = np.empty_like(x64)
    b0, b1, b2 = b
    _, a1, a2 = a

    x0 = x64[..., 0]
    z1 = (1.0 - b0) * x0
    z2 = (1.0 - b0 - b1 + a1) * x0
    for i in range(x64.shape[-1]):
        xi = x64[..., i]
        yi = b0 * xi + z1
        y[..., i] = yi
        z1_next = b1 * xi - a1 * yi + z2
        z2 = b2 * xi - a2 * yi
        z1 = z1_next
    return y


def _filtfilt_biquad(x, b, a):
    y = _lfilter_biquad(x, b, a)
    y = np.flip(_lfilter_biquad(np.flip(y, axis=-1), b, a), axis=-1)
    return y


def apply_emg_notch_filters(
    emg,
    fs=EMG_FS,
    freqs=EMG_NOTCH_FREQS,
    half_width_hz=EMG_NOTCH_HALF_WIDTH_HZ,
):
    """Apply zero-phase 150/250 Hz notch filters on raw EMG arrays shaped (..., time)."""
    try:
        from scipy.signal import filtfilt, iirnotch

        filtered = emg.astype(np.float64, copy=False)
        for freq in freqs:
            q = freq / (2.0 * half_width_hz)
            try:
                b, a = iirnotch(freq, q, fs=fs)
            except TypeError:
                b, a = iirnotch(freq / (fs * 0.5), q)
            filtered = filtfilt(b, a, filtered, axis=-1)
    except ImportError:
        filtered = emg
        for freq in freqs:
            b, a = _notch_coeffs(freq, fs, half_width_hz)
            filtered = _filtfilt_biquad(filtered, b, a)

    return np.ascontiguousarray(filtered, dtype=np.float32)


def load_data_splits(
    base_dir,
    paradigms=('overt', 'silent'),
    unseen_subjects=('S28', 'S29', 'S30'),
    train_subjects=None,
    val_ratio=0.1,
    emg_notch=True,
    emg_fs=EMG_FS,
    emg_notch_freqs=EMG_NOTCH_FREQS,
    emg_notch_half_width_hz=EMG_NOTCH_HALF_WIDTH_HZ,
):
    """Load processed EEG, raw EMG, labels, and Overt acoustic features.

    Expected layout:
        base_dir/{paradigm}/S01/
            S01_eeg_feat.npy    (N, 60, 256)
            S01_emg.npy         (N, 6, >=1000), first 1000 samples used
            S01_audio_feat.npy  (N, 50, 768), HuBERT block-9 frames, Overt only
            S01_labels.npy      (N,), source labels 1-10

    Unseen subjects form ``val_c``. For other selected subjects, the first
    ``1 - val_ratio`` trials form training data and the remainder form
    ``val_w`` in temporal order.
    """
    print("\n" + "=" * 55)
    print(f" Loading EEG/EMG data: paradigms={list(paradigms)}")
    if train_subjects is not None:
        print(f" Selected training subjects: {sorted(train_subjects)}")
    if emg_notch:
        print(
            f" EMG notch: fs={emg_fs:g}Hz, freqs={tuple(emg_notch_freqs)}, "
            f"half_width={emg_notch_half_width_hz:g}Hz"
        )
    print("=" * 55)

    keys = ('eeg', 'emg', 'audio', 'labels', 'is_overt')
    splits = {s: {k: [] for k in keys} for s in ('train', 'val_w', 'val_c')}
    total_loaded = 0

    unseen_set = set(unseen_subjects)
    train_set  = set(train_subjects) if train_subjects is not None else None

    for paradigm in paradigms:
        is_overt_val = np.float32(1.0 if paradigm == 'overt' else 0.0)
        para_dir = os.path.join(base_dir, paradigm)

        subject_dirs = sorted(glob.glob(os.path.join(para_dir, 'S*')))
        if not subject_dirs:
            print(f"  [WARN] No subject directories found: {para_dir}/S*")
            continue

        for subj_dir in subject_dirs:
            subj_id = os.path.basename(subj_dir)

            if subj_id in unseen_set:
                dest = 'val_c'
            elif train_set is not None and subj_id not in train_set:
                continue
            else:
                dest = 'train_val_w'

            eeg_path   = os.path.join(subj_dir, f'{subj_id}_eeg_feat.npy')
            emg_path   = os.path.join(subj_dir, f'{subj_id}_emg.npy')
            audio_path = os.path.join(subj_dir, f'{subj_id}_audio_feat.npy')
            label_path = os.path.join(subj_dir, f'{subj_id}_labels.npy')

            required = [eeg_path, emg_path, label_path]
            if paradigm == 'overt':
                required.append(audio_path)

            if not all(os.path.exists(p) for p in required):
                print(f"  [SKIP] {paradigm}/{subj_id}: missing required file")
                continue

            eeg_raw = np.load(eeg_path)
            emg_raw = np.load(emg_path)
            if emg_raw.ndim == 3 and emg_raw.shape[1] != 6 and emg_raw.shape[2] == 6:
                emg_raw = np.transpose(emg_raw, (0, 2, 1))
            labels_raw = np.load(label_path)

            if paradigm == 'overt':
                audio_raw = np.load(audio_path)
                _validate_trial_alignment(
                    eeg_raw, emg_raw, audio_raw, labels_raw,
                    eeg_path, emg_path, audio_path, label_path,
                )
                # Average exactly across the retained 50-frame temporal axis.
                audio_data = audio_raw.mean(axis=1, dtype=np.float32)
            else:
                if eeg_raw.ndim != 3 or eeg_raw.shape[1:] != (60, 256):
                    raise ValueError(f'{eeg_path}: expected EEG shape (N, 60, 256), got {eeg_raw.shape}')
                if emg_raw.ndim != 3 or emg_raw.shape[1] != 6 or emg_raw.shape[-1] < EMG_RAW_LEN:
                    raise ValueError(f'{emg_path}: expected EMG shape (N, 6, >= {EMG_RAW_LEN}), got {emg_raw.shape}')
                if labels_raw.ndim != 1:
                    raise ValueError(f'{label_path}: expected labels shape (N,), got {labels_raw.shape}')
                for name, array, path in (
                    ('EEG', eeg_raw, eeg_path), ('EMG', emg_raw, emg_path), ('labels', labels_raw, label_path)
                ):
                    _require_finite(name, array, path)
                if len({len(eeg_raw), len(emg_raw), len(labels_raw)}) != 1:
                    raise ValueError('positional trial alignment requires equal EEG, EMG, and label trial counts')
                audio_data = np.zeros((len(labels_raw), 768), dtype=np.float32)

            eeg_data = np.ascontiguousarray(eeg_raw, dtype=np.float32)
            emg_data = np.ascontiguousarray(emg_raw[:, :, :EMG_RAW_LEN], dtype=np.float32)
            if emg_notch:
                emg_data = apply_emg_notch_filters(
                    emg_data, fs=emg_fs, freqs=emg_notch_freqs, half_width_hz=emg_notch_half_width_hz,
                )
            labels = labels_raw.astype(np.int64) - 1  # 0-indexed

            n        = len(labels)
            is_overt = np.full(n, is_overt_val, dtype=np.float32)
            total_loaded += 1

            arrays = [eeg_data, emg_data, audio_data, labels, is_overt]

            if dest == 'val_c':
                for arr, key in zip(arrays, keys):
                    splits['val_c'][key].append(arr)
            else:
                n_train = int(n * (1.0 - val_ratio))
                for arr, key in zip(arrays, keys):
                    splits['train'][key].append(arr[:n_train])
                    splits['val_w'][key].append(arr[n_train:])

    def _concat(split_name):
        d = splits[split_name]
        if not d['labels']:
            return {
                'eeg':      np.zeros((0, 60, 256), dtype=np.float32),
                'emg':      np.zeros((0, 6,  EMG_RAW_LEN), dtype=np.float32),
                'audio':    np.zeros((0, 768),      dtype=np.float32),
                'labels':   np.zeros((0,),           dtype=np.int64),
                'is_overt': np.zeros((0,),           dtype=np.float32),
            }
        return {k: np.concatenate(d[k], axis=0) for k in keys}

    train_data = _concat('train')
    val_w_data = _concat('val_w')
    val_c_data = _concat('val_c')

    print(f"  Loaded {total_loaded} subject/paradigm combinations")
    for name, d in [('train', train_data), ('val_w', val_w_data), ('val_c', val_c_data)]:
        n = len(d['labels'])
        if n > 0:
            overt_n = int(d['is_overt'].sum())
            print(f"  {name:6s}: {n:5d} trials (overt={overt_n}, silent={n-overt_n})")
        else:
            print(f"  {name:6s}: 0 trials")
    print("=" * 55 + "\n")

    return train_data, val_w_data, val_c_data


class BCIDataset(Dataset):
    """Dataset returning EEG, EMG, audio, labels, and Overt indicators."""

    def __init__(self, data_dict):
        self.eeg      = torch.from_numpy(data_dict['eeg']).float()
        self.emg      = torch.from_numpy(data_dict['emg']).float()
        self.audio    = torch.from_numpy(data_dict['audio']).float()
        self.labels   = torch.from_numpy(data_dict['labels']).long()
        self.is_overt = torch.from_numpy(data_dict['is_overt']).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.eeg[idx], self.emg[idx], self.audio[idx],
                self.labels[idx], self.is_overt[idx])
