import numpy as np
import pytest

from privileged_emg.dataset import _validate_trial_alignment


def _arrays():
    return (
        np.zeros((2, 60, 256), dtype=np.float32),
        np.zeros((2, 6, 1000), dtype=np.float32),
        np.zeros((2, 50, 768), dtype=np.float32),
        np.array([1, 2], dtype=np.int64),
    )


def _validate(eeg, emg, audio, labels):
    _validate_trial_alignment(eeg, emg, audio, labels, 'eeg.npy', 'emg.npy', 'audio.npy', 'labels.npy')


def test_audio_feature_contract_accepts_aligned_float32_finite_arrays():
    _validate(*_arrays())


@pytest.mark.parametrize(
    ('mutate', 'message'),
    [
        (lambda eeg, emg, audio, labels: (eeg, emg, audio[:, :49], labels), 'audio shape'),
        (lambda eeg, emg, audio, labels: (eeg, emg, audio.astype(np.float64), labels), 'audio dtype'),
        (lambda eeg, emg, audio, labels: (eeg, emg, np.full_like(audio, np.nan), labels), 'finite'),
        (lambda eeg, emg, audio, labels: (eeg[:1], emg, audio, labels), 'equal trial counts'),
    ],
)
def test_audio_feature_contract_rejects_invalid_or_misaligned_arrays(mutate, message):
    with pytest.raises(ValueError, match=message):
        _validate(*mutate(*_arrays()))
