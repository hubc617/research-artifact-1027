import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from privileged_emg.label_simplex import (
    EEGIndexedDataset,
    FinalTestBoundary,
    LabelSimplexPretrainer,
    PROTOTYPE_NPY_SHA256,
    assert_eeg_student_initialization_identity,
    assert_eeg_only_path,
    label_targets,
    load_eeg_label_cell,
    prototype_npy_bytes,
    regular_simplex_prototypes,
)
from privileged_emg.model_stage2 import EEGMAEAlignerPretrainer


ROOT = Path(__file__).resolve().parents[1]


def test_ten_simplex_vectors_are_deterministic_and_frozen() -> None:
    first = regular_simplex_prototypes()
    second = regular_simplex_prototypes()
    assert np.array_equal(first, second)
    assert first.shape == (10, 256)
    assert np.array_equal(first[:, 10:], np.zeros((10, 246), dtype=np.float32))
    assert hashlib.sha256(prototype_npy_bytes()).hexdigest() == PROTOTYPE_NPY_SHA256


def test_target_assignment_depends_only_on_lexical_label() -> None:
    prototypes = torch.from_numpy(regular_simplex_prototypes())
    labels = torch.tensor([3, 1, 3, 9], dtype=torch.long)
    targets = label_targets(labels, prototypes)
    assert torch.equal(targets[0], targets[2])
    assert torch.equal(targets, prototypes.index_select(0, labels))


def test_eeg_only_loader_opens_no_emg_trial_or_checkpoint(tmp_path, monkeypatch) -> None:
    cell = tmp_path / "overt" / "S01"
    cell.mkdir(parents=True)
    np.save(cell / "S01_eeg_feat.npy", np.zeros((2, 60, 256), dtype=np.float32))
    np.save(cell / "S01_labels.npy", np.array([1, 10], dtype=np.int64))
    opened = []
    original = np.load

    def recording_load(path, *args, **kwargs):
        opened.append(Path(path).name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", recording_load)
    eeg, labels = load_eeg_label_cell(tmp_path, "S01", "overt")
    assert eeg.shape == (2, 60, 256)
    assert labels.tolist() == [0, 9]
    assert opened == ["S01_eeg_feat.npy", "S01_labels.npy"]
    assert not any("emg" in name.lower() or "stage1" in name.lower() for name in opened)
    indexed = EEGIndexedDataset(
        cell / "S01_eeg_feat.npy", cell / "S01_labels.npy",
        np.array([1], dtype=np.int64), "final_test",
    )
    assert len(indexed) == 1
    assert opened[-2:] == ["S01_eeg_feat.npy", "S01_labels.npy"]
    with pytest.raises(PermissionError):
        assert_eeg_only_path(tmp_path / "overt/S01/S01_emg.npy")
    with pytest.raises(PermissionError):
        assert_eeg_only_path(tmp_path / "stage1/best_stage1.pth")


def test_runtime_emg_audit_hook_fails_closed() -> None:
    code = (
        "from privileged_emg.label_simplex import install_emg_access_guard; "
        "install_emg_access_guard(); open('/tmp/forbidden_emg.npy', 'rb')"
    )
    run = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True)
    assert run.returncode != 0
    assert "LabelSimplex EMG-access guard blocked" in run.stderr


def test_eeg_trainable_initialization_matches_clean_full(monkeypatch) -> None:
    monkeypatch.setattr(EEGMAEAlignerPretrainer, "_load_emg_encoder", lambda self, path: None)
    prototypes = torch.from_numpy(regular_simplex_prototypes())
    torch.manual_seed(42)
    full = EEGMAEAlignerPretrainer("not_opened", 256, 10, 0.75, weak_weight=0.3)
    torch.manual_seed(42)
    control = LabelSimplexPretrainer(prototypes, 256, 10, 0.75)
    assert_eeg_student_initialization_identity(full, control)


def test_final_test_cannot_affect_selection(tmp_path) -> None:
    checkpoint = tmp_path / "best_stage3.pth"
    boundary = FinalTestBoundary(checkpoint)
    assert boundary.consider_validation(40.0, 1)
    assert not boundary.consider_validation(39.0, 2)
    with pytest.raises(RuntimeError):
        boundary.authorize_final_test_access()
    checkpoint.write_bytes(b"validation-selected")
    boundary.freeze_selection()
    with pytest.raises(RuntimeError):
        boundary.consider_validation(99.0, 3)
    boundary.mark_selected_checkpoint_reloaded(checkpoint)
    boundary.authorize_final_test_access()
    with pytest.raises(RuntimeError):
        boundary.authorize_final_test_access()
    assert boundary.best_validation_balanced_accuracy == 40.0
    assert boundary.audit()["test_used_for_selection"] is False


def test_label_simplex_manifests_are_byte_identical(tmp_path) -> None:
    script = ROOT / "scripts/generate_protocol_manifests.py"
    common = [
        sys.executable, str(script),
        "--data-root", str(tmp_path / "data"),
        "--split-root", str(tmp_path / "splits"),
        "--run-root", str(tmp_path / "runs"),
    ]
    first, second = tmp_path / "first", tmp_path / "second"
    subprocess.run([*common, "--output-dir", str(first)], check=True, capture_output=True)
    subprocess.run([*common, "--output-dir", str(second)], check=True, capture_output=True)
    names = ("label_simplex_stage2_15.jsonl", "label_simplex_stage3_900.jsonl")
    for name in names:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert len((first / names[0]).read_text().splitlines()) == 15
    assert len((first / names[1]).read_text().splitlines()) == 900
