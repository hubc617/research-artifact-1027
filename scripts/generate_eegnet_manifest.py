"""Preflight the recovered 4090 data/splits and freeze the 900-task manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SEEDS = (42, 2026, 3407)
SUBJECTS = tuple(f"S{i:02d}" for i in range(1, 31))
PARADIGMS = ("overt", "silent_1001")


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    data = args.data_root.resolve()
    split_root = args.split_root.resolve()
    run_root = args.run_root.resolve()
    manifest_dir = args.output_dir.resolve()
    manifest_dir.mkdir(parents=True, exist_ok=True)

    for required in (data, split_root):
        if not required.exists():
            raise FileNotFoundError(required)

    protocol = {
        "baseline": "EEGNet-8,2",
        "reference_doi": "10.1088/1741-2552/aace8c",
        "input": "EEG only; 60 channels x 256 samples",
        "subjects": list(SUBJECTS),
        "paradigms": list(PARADIGMS),
        "folds": list(range(5)),
        "seeds": list(SEEDS),
        "epochs": 100,
        "batch_size": 8,
        "eval_batch_size": 64,
        "optimizer": "Adam",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "scheduler": "CosineAnnealingLR",
        "eta_min": 1e-6,
        "label_smoothing": 0.1,
        "gradient_clip": 1.0,
        "checkpoint_selection": "maximum validation balanced accuracy",
        "test_policy": "evaluate once after validation checkpoint selection",
        "no_privileged_inputs": ["EMG", "audio", "acoustic features", "Stage-1 checkpoint", "Stage-2 checkpoint"],
    }

    tasks = []
    trial_counts = {}
    for subject in SUBJECTS:
        for paradigm in PARADIGMS:
            subject_dir = data / paradigm / subject
            eeg_path = subject_dir / f"{subject}_eeg_feat.npy"
            label_path = subject_dir / f"{subject}_labels.npy"
            if not eeg_path.is_file() or not label_path.is_file():
                raise FileNotFoundError(f"Missing EEG/labels for {paradigm}/{subject}")
            eeg = np.load(eeg_path, mmap_mode="r")
            labels = np.load(label_path, mmap_mode="r")
            if eeg.ndim != 3 or tuple(eeg.shape[1:]) != (60, 256):
                raise ValueError(f"{eeg_path}: expected (N,60,256), got {eeg.shape}")
            if len(eeg) != len(labels):
                raise ValueError(f"{eeg_path}: EEG/label length mismatch")
            trial_counts[f"{subject}/{paradigm}"] = int(len(labels))

            for fold in range(5):
                split_path = split_root / f"{subject}_{paradigm}_fold{fold}.npz"
                if not split_path.is_file():
                    raise FileNotFoundError(split_path)
                split = np.load(split_path)
                tr = np.asarray(split["train_idx"], dtype=np.int64)
                va = np.asarray(split["val_idx"], dtype=np.int64)
                te = np.asarray(split["test_idx"], dtype=np.int64)
                if set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te):
                    raise RuntimeError(f"Leakage in {split_path}")
                if len(tr) + len(va) + len(te) != len(labels):
                    raise RuntimeError(f"Incomplete partition in {split_path}")

    for seed in SEEDS:
        for fold in range(5):
            for subject in SUBJECTS:
                for paradigm in PARADIGMS:
                    subject_dir = data / paradigm / subject
                    out_dir = run_root / f"seed{seed}" / f"fold{fold}" / subject / paradigm
                    tasks.append({
                        "baseline": "eegnet_8_2",
                        "seed": seed,
                        "fold": fold,
                        "subject": subject,
                        "paradigm": paradigm,
                        "epochs": protocol["epochs"],
                        "batch_size": protocol["batch_size"],
                        "eval_batch_size": protocol["eval_batch_size"],
                        "learning_rate": protocol["learning_rate"],
                        "weight_decay": protocol["weight_decay"],
                        "eta_min": protocol["eta_min"],
                        "label_smoothing": protocol["label_smoothing"],
                        "gradient_clip": protocol["gradient_clip"],
                        "eeg_path": str(subject_dir / f"{subject}_eeg_feat.npy"),
                        "label_path": str(subject_dir / f"{subject}_labels.npy"),
                        "split_file": str(split_root / f"{subject}_{paradigm}_fold{fold}.npz"),
                        "out_dir": str(out_dir),
                    })

    if len(tasks) != 900:
        raise RuntimeError(f"Expected 900 tasks, generated {len(tasks)}")
    manifest_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        for row in tasks
    )
    manifest_path = manifest_dir / "eegnet_3seed_900tasks.jsonl"
    if manifest_path.exists() and manifest_path.read_bytes() != manifest_bytes:
        raise RuntimeError(
            f"Refusing to overwrite a different frozen manifest: {manifest_path}"
        )
    manifest_path.write_bytes(manifest_bytes)
    protocol["data_root"] = str(data)
    protocol["split_root"] = str(split_root)
    protocol["run_root"] = str(run_root)
    protocol["manifest"] = str(manifest_path)
    protocol["manifest_sha256"] = digest_bytes(manifest_bytes)
    protocol["task_count"] = len(tasks)
    protocol["trial_counts"] = trial_counts
    (manifest_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False)
    )
    (manifest_dir / "manifest.sha256").write_text(
        f"{protocol['manifest_sha256']}  {manifest_path.name}\n"
    )
    print(json.dumps(protocol, indent=2, ensure_ascii=False))
    print("EEGNET_MANIFEST_PREFLIGHT_OK")


if __name__ == "__main__":
    main()
