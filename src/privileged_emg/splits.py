"""Deterministic subject-dependent, trial-level stratified five-fold splits."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from .common import load_unseen_subjects

SUBJECTS = [f"S{i:02d}" for i in range(1, 31)]
PARADIGMS = ("overt", "silent_1001")
OUTER_RANDOM_STATE = 20260811
VAL_RANDOM_BASE = 20261811


def generate_splits(output: str | Path) -> Path:
    """Write 300 split files without overwriting an existing output directory."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for subject in SUBJECTS:
        for paradigm in PARADIGMS:
            labels = np.asarray(load_unseen_subjects([subject], [paradigm])["labels"])
            folds = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=OUTER_RANDOM_STATE
            ).split(np.zeros(len(labels)), labels)
            for fold, (pool_idx, test_idx) in enumerate(folds):
                paradigm_offset = 0 if paradigm == "overt" else 100000
                val_seed = VAL_RANDOM_BASE + 1000 * fold + 10 * int(subject[1:]) + paradigm_offset
                train_local, val_local = next(StratifiedShuffleSplit(
                    n_splits=1, test_size=0.15, random_state=val_seed
                ).split(np.zeros(len(pool_idx)), labels[pool_idx]))
                train_idx, val_idx = pool_idx[train_local], pool_idx[val_local]
                if set(train_idx) & set(val_idx) or set(train_idx) & set(test_idx) or set(val_idx) & set(test_idx):
                    raise AssertionError(f"overlapping partitions: {subject}/{paradigm}/fold{fold}")
                filename = f"{subject}_{paradigm}_fold{fold}.npz"
                np.savez_compressed(output / filename, train_idx=train_idx.astype(np.int64), val_idx=val_idx.astype(np.int64), test_idx=test_idx.astype(np.int64))
                manifest.append({
                    "subject": subject, "paradigm": paradigm, "fold": fold,
                    "n_total": len(labels), "n_train": len(train_idx),
                    "n_val": len(val_idx), "n_test": len(test_idx), "split_file": filename,
                })
    if len(manifest) != 300:
        raise AssertionError(f"expected 300 split entries, found {len(manifest)}")
    manifest_path = output / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "manifest_sha256.txt").write_text(hashlib.sha256(manifest_path.read_bytes()).hexdigest() + "\n")
    return manifest_path
