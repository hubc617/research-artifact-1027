"""EEG-only LabelSimplex control used by the clean-20260811 protocol."""
from __future__ import annotations

import hashlib
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from .model_MAE import MAE_Decoder, MAE_Encoder


SUBJECTS = tuple(f"S{i:02d}" for i in range(1, 31))
PARADIGMS = ("overt", "silent_1001")
PROTOTYPE_NPY_SHA256 = "1f0698d642af6bf2a219a0d704adbce239e9bca5c12e89f37484581ad68bf3c1"


def regular_simplex_prototypes() -> np.ndarray:
    """Return the frozen 10-class regular simplex embedded in R^256."""
    centered = np.eye(10, dtype=np.float64) - np.ones((10, 10), dtype=np.float64) / 10.0
    centered /= np.linalg.norm(centered, axis=1, keepdims=True)
    prototypes = np.zeros((10, 256), dtype="<f4")
    prototypes[:, :10] = centered.astype("<f4")
    validate_prototypes(prototypes)
    return prototypes


def prototype_npy_bytes() -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, regular_simplex_prototypes(), allow_pickle=False)
    return buffer.getvalue()


def validate_prototypes(array: np.ndarray) -> None:
    if array.shape != (10, 256) or array.dtype != np.dtype("<f4"):
        raise AssertionError("LabelSimplex prototype shape/dtype mismatch")
    if not np.array_equal(array[:, 10:], np.zeros((10, 246), dtype="<f4")):
        raise AssertionError("LabelSimplex prototype zero padding mismatch")
    if not np.allclose(np.linalg.norm(array, axis=1), 1.0, rtol=0.0, atol=1e-7):
        raise AssertionError("LabelSimplex prototype rows are not unit norm")
    gram = array @ array.T
    expected = np.full((10, 10), -1.0 / 9.0)
    np.fill_diagonal(expected, 1.0)
    if not np.allclose(gram, expected, rtol=0.0, atol=3e-7):
        raise AssertionError("LabelSimplex prototype Gram matrix is invalid")
    if not np.allclose(array.sum(axis=0), 0.0, rtol=0.0, atol=2e-7):
        raise AssertionError("LabelSimplex prototype centroid is not zero")
    digest = hashlib.sha256(prototype_npy_bytes_unchecked(array)).hexdigest()
    if digest != PROTOTYPE_NPY_SHA256:
        raise AssertionError(f"LabelSimplex prototype hash mismatch: {digest}")


def prototype_npy_bytes_unchecked(array: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return buffer.getvalue()


def label_targets(labels: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    if labels.ndim != 1 or labels.dtype != torch.long:
        raise AssertionError("LabelSimplex labels must be a one-dimensional int64 tensor")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= 10):
        raise AssertionError("LabelSimplex label is outside [0, 9]")
    return prototypes.index_select(0, labels)


def is_forbidden_emg_path(path: str | os.PathLike[str]) -> bool:
    resolved = Path(path).resolve(strict=False)
    name = resolved.name.lower()
    parts = tuple(part.lower() for part in resolved.parts)
    return (
        name.endswith("_emg.npy")
        or "best_stage1" in name
        or "emg_encoder" in name
        or "stage1" in parts
    )


def assert_eeg_only_path(path: str | os.PathLike[str]) -> None:
    if is_forbidden_emg_path(path):
        raise PermissionError(f"LabelSimplex EMG-access guard blocked: {path}")


def install_emg_access_guard() -> None:
    """Install a process-wide fail-closed guard before loading Stage-2 data."""
    def hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            assert_eeg_only_path(os.fsdecode(args[0]))

    sys.addaudithook(hook)


class EEGLabelDataset(Dataset):
    def __init__(self, eeg: np.ndarray, labels: np.ndarray):
        self.eeg = torch.from_numpy(np.ascontiguousarray(eeg, dtype=np.float32))
        self.labels = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))
        if self.eeg.ndim != 3 or self.eeg.shape[1:] != (60, 256):
            raise AssertionError(f"Malformed EEG array: {tuple(self.eeg.shape)}")
        if self.labels.ndim != 1 or len(self.eeg) != len(self.labels):
            raise AssertionError("EEG/label trial identity mismatch")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.eeg[index], self.labels[index]


class EEGIndexedDataset(Dataset):
    """Materialize only the indexed EEG/label partition for Stage 3."""
    def __init__(self, eeg_path: str | Path, label_path: str | Path,
                 indices: np.ndarray, role: str):
        if role not in {"train", "validation", "final_test"}:
            raise AssertionError(f"Unknown dataset role: {role}")
        assert_eeg_only_path(eeg_path)
        assert_eeg_only_path(label_path)
        eeg = np.load(eeg_path, mmap_mode="r", allow_pickle=False)
        labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
        indices = np.asarray(indices, dtype=np.int64)
        if eeg.shape[0] != labels.shape[0]:
            raise AssertionError("EEG/label trial identity mismatch")
        if indices.ndim != 1 or not len(indices):
            raise AssertionError(f"Empty or malformed {role} indices")
        if indices.min() < 0 or indices.max() >= len(labels):
            raise AssertionError(f"Out-of-range {role} index")
        self.eeg = torch.from_numpy(np.asarray(eeg[indices], dtype=np.float32))
        self.labels = torch.from_numpy(
            np.asarray(labels[indices], dtype=np.int64) - 1
        )
        if self.eeg.ndim != 3 or tuple(self.eeg.shape[1:]) != (60, 256):
            raise AssertionError(f"Malformed EEG array: {tuple(self.eeg.shape)}")
        if self.labels.min() < 0 or self.labels.max() >= 10:
            raise AssertionError("Labels must map to [0, 9]")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.eeg[index], self.labels[index]


def load_partition_indices(
    split_root: str | Path,
    subject: str,
    paradigm: str,
    fold: int,
    role: str,
) -> np.ndarray:
    keys = {"train": "train_idx", "validation": "val_idx", "final_test": "test_idx"}
    if role not in keys:
        raise AssertionError(f"Unknown split role: {role}")
    split_path = Path(split_root) / f"{subject}_{paradigm}_fold{fold}.npz"
    assert_eeg_only_path(split_path)
    with np.load(split_path, allow_pickle=False) as split:
        return np.asarray(split[keys[role]], dtype=np.int64)


def stage3_data_paths(
    data_root: str | Path,
    subject: str,
    paradigm: str,
) -> tuple[Path, Path]:
    directory = Path(data_root) / paradigm / subject
    paths = (
        directory / f"{subject}_eeg_feat.npy",
        directory / f"{subject}_labels.npy",
    )
    for path in paths:
        assert_eeg_only_path(path)
    return paths


def load_eeg_label_cell(
    data_root: str | Path,
    subject: str,
    paradigm: str,
) -> tuple[np.ndarray, np.ndarray]:
    directory = Path(data_root) / paradigm / subject
    eeg_path = directory / f"{subject}_eeg_feat.npy"
    label_path = directory / f"{subject}_labels.npy"
    assert_eeg_only_path(eeg_path)
    assert_eeg_only_path(label_path)
    eeg = np.load(eeg_path, allow_pickle=False).astype(np.float32, copy=False)
    labels = np.load(label_path, allow_pickle=False).astype(np.int64, copy=False) - 1
    if eeg.shape != (len(labels), 60, 256):
        raise AssertionError(f"EEG/label identity mismatch: {subject}/{paradigm}")
    if labels.size and (labels.min() < 0 or labels.max() > 9):
        raise AssertionError(f"Label range mismatch: {subject}/{paradigm}")
    return eeg, labels


def load_eeg_label_train_validation(
    data_root: str | Path,
    split_root: str | Path,
    fold: int,
    subjects: tuple[str, ...] | list[str] = SUBJECTS,
) -> tuple[EEGLabelDataset, EEGLabelDataset, dict]:
    train_eeg, train_labels, val_eeg, val_labels = [], [], [], []
    identity = hashlib.sha256()
    subjects = tuple(subjects)
    if not subjects or len(set(subjects)) != len(subjects):
        raise AssertionError("subjects must be nonempty and unique")
    if not set(subjects).issubset(SUBJECTS):
        raise AssertionError("unknown subject in EEG-only Stage-2 request")
    for subject in subjects:
        for paradigm in PARADIGMS:
            eeg, labels = load_eeg_label_cell(data_root, subject, paradigm)
            split_path = Path(split_root) / f"{subject}_{paradigm}_fold{fold}.npz"
            assert_eeg_only_path(split_path)
            with np.load(split_path, allow_pickle=False) as split:
                train_idx = np.asarray(split["train_idx"], dtype=np.int64)
                val_idx = np.asarray(split["val_idx"], dtype=np.int64)
            if np.intersect1d(train_idx, val_idx).size:
                raise AssertionError(f"Train/validation overlap: {split_path}")
            if max(train_idx.max(), val_idx.max()) >= len(labels):
                raise AssertionError(f"Split index out of range: {split_path}")
            train_eeg.append(eeg[train_idx])
            train_labels.append(labels[train_idx])
            val_eeg.append(eeg[val_idx])
            val_labels.append(labels[val_idx])
            for name, values in (("train", train_idx), ("validation", val_idx)):
                identity.update(subject.encode())
                identity.update(paradigm.encode())
                identity.update(name.encode())
                identity.update(values.astype("<i8", copy=False).tobytes())
    train = EEGLabelDataset(np.concatenate(train_eeg), np.concatenate(train_labels))
    validation = EEGLabelDataset(np.concatenate(val_eeg), np.concatenate(val_labels))
    audit = {
        "loader": "EEG_AND_LABELS_ONLY",
        "emg_trial_files_opened": 0,
        "emg_checkpoints_opened": 0,
        "fold": int(fold),
        "n_subject_paradigm_cells": 2 * len(subjects),
        "subjects": list(subjects),
        "n_train": len(train),
        "n_validation": len(validation),
        "ordered_train_validation_identity_sha256": identity.hexdigest(),
    }
    return train, validation, audit


class LabelSimplexPretrainer(nn.Module):
    """Clean Full EEG student with a fixed label-only representation target."""
    def __init__(self, prototypes: torch.Tensor, shared_dim: int = 256,
                 num_classes: int = 10, eeg_mask_ratio: float = 0.75):
        super().__init__()
        self.eeg_mask_ratio = eeg_mask_ratio
        self.eeg_encoder = MAE_Encoder(
            in_channels=60, embed_dim=shared_dim, patch_size=16, seq_len=256
        )
        self.eeg_decoder = MAE_Decoder(
            in_channels=60, embed_dim=shared_dim, patch_size=16,
            seq_len=256, is_eeg=True
        )
        self.eeg_proj = nn.Sequential(
            nn.Linear(shared_dim, shared_dim),
            nn.GELU(),
            nn.Linear(shared_dim, shared_dim),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(shared_dim),
            nn.Dropout(0.3),
            nn.Linear(shared_dim, num_classes),
        )
        if tuple(prototypes.shape) != (10, 256):
            raise AssertionError("Expected fixed 10x256 LabelSimplex prototypes")
        self.register_buffer(
            "label_prototypes",
            prototypes.detach().to(dtype=torch.float32).clone(),
            persistent=True,
        )
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)

    @staticmethod
    def _mae_loss(original, prediction, mask, patch_size=16):
        original = original[:, :, :prediction.shape[-1]]
        per_time = ((prediction - original) ** 2).mean(dim=1)
        expanded = mask.repeat_interleave(patch_size, dim=1)
        return (per_time * expanded).sum() / (expanded.sum() + 1e-8)

    def forward(self, eeg, labels, use_label_loss: bool):
        eeg = (eeg - eeg.mean(dim=-1, keepdim=True)) / (
            eeg.std(dim=-1, keepdim=True) + 1e-8
        )
        visible, mask, restore = self.eeg_encoder(eeg, self.eeg_mask_ratio)
        reconstruction = self.eeg_decoder(visible, restore)
        loss_mae = self._mae_loss(eeg, reconstruction, mask)
        cls = visible[:, 0, :]
        z_eeg = self.eeg_proj(cls)
        logits = self.classifier(cls)
        loss_ce = self.ce_loss(logits, labels)
        if use_label_loss:
            targets = label_targets(labels, self.label_prototypes)
            loss_label = (
                1.0 - (F.normalize(z_eeg, dim=-1) * targets).sum(dim=-1)
            ).mean()
        else:
            loss_label = torch.zeros((), device=eeg.device)
        return {
            "loss_mae": loss_mae,
            "loss_label": loss_label,
            "loss_ce": loss_ce,
            "logits": logits,
        }


def assert_eeg_student_initialization_identity(
    clean_full: nn.Module,
    label_simplex: LabelSimplexPretrainer,
) -> None:
    """Require bitwise identity for all trainable EEG components."""
    for name in ("eeg_encoder", "eeg_decoder", "eeg_proj", "classifier"):
        expected = getattr(clean_full, name).state_dict()
        actual = getattr(label_simplex, name).state_dict()
        if expected.keys() != actual.keys():
            raise AssertionError(f"EEG initialization keys differ: {name}")
        changed = [key for key in expected if not torch.equal(expected[key], actual[key])]
        if changed:
            raise AssertionError(f"EEG initialization differs: {name}.{changed[0]}")


@dataclass
class FinalTestBoundary:
    """State machine proving validation selection precedes final-test access."""
    checkpoint: Path
    best_validation_balanced_accuracy: float = -1.0
    best_epoch: int = 0
    selection_frozen: bool = False
    selected_checkpoint_reloaded: bool = False
    test_accesses: int = 0

    def consider_validation(self, balanced_accuracy: float, epoch: int) -> bool:
        if self.selection_frozen:
            raise RuntimeError("Checkpoint selection is already frozen")
        if balanced_accuracy > self.best_validation_balanced_accuracy:
            self.best_validation_balanced_accuracy = float(balanced_accuracy)
            self.best_epoch = int(epoch)
            return True
        return False

    def freeze_selection(self) -> None:
        if self.best_epoch < 1 or not self.checkpoint.is_file():
            raise RuntimeError("Cannot freeze without a validation-selected checkpoint")
        self.selection_frozen = True

    def mark_selected_checkpoint_reloaded(self, checkpoint: str | Path) -> None:
        if not self.selection_frozen or Path(checkpoint).resolve() != self.checkpoint.resolve():
            raise RuntimeError("Final-test boundary requires the frozen checkpoint reload")
        self.selected_checkpoint_reloaded = True

    def authorize_final_test_access(self) -> None:
        if not self.selection_frozen or not self.selected_checkpoint_reloaded:
            raise RuntimeError("Final-test access requested before selection and reload")
        if self.test_accesses:
            raise RuntimeError("Final-test access is permitted exactly once")
        self.test_accesses += 1

    def audit(self) -> dict:
        return {
            "selection_metric": "validation_balanced_accuracy",
            "selection_frozen_before_test": self.selection_frozen,
            "selected_checkpoint_reloaded_before_test": self.selected_checkpoint_reloaded,
            "test_loader_constructions": self.test_accesses,
            "test_used_for_selection": False,
        }
