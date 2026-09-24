#!/usr/bin/env python3
"""Generate deterministic subject-isolated Stage 1/2/3 manifests.

The six target subjects assigned to each outer fold are excluded from every
upstream Stage-1 and Stage-2 task.  Target train/validation/final-test indices
are exported into separate files so final-test indices need not be opened
during model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SEEDS = (42, 2026, 3407)
FOLDS = tuple(range(5))
SUBJECTS = tuple(f"S{i:02d}" for i in range(1, 31))
PARADIGMS = ("overt", "silent_1001")
VARIANTS = ("full", "no_distill")
TARGET_FOLDS = {
    0: ("S09", "S12", "S13", "S15", "S24", "S26"),
    1: ("S08", "S20", "S21", "S22", "S23", "S29"),
    2: ("S04", "S05", "S07", "S10", "S11", "S27"),
    3: ("S01", "S02", "S16", "S17", "S25", "S30"),
    4: ("S03", "S06", "S14", "S18", "S19", "S28"),
}


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_new(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> str:
    return write_new(path, b"".join(canonical_bytes(row) for row in rows))


def export_split_roles(source_root: Path, output_root: Path) -> dict[tuple[str, str, int], dict]:
    records = {}
    for subject in SUBJECTS:
        for paradigm in PARADIGMS:
            for fold in FOLDS:
                source = source_root / f"{subject}_{paradigm}_fold{fold}.npz"
                with np.load(source, allow_pickle=False) as split:
                    arrays = {
                        "train": np.asarray(split["train_idx"], dtype=np.int64),
                        "validation": np.asarray(split["val_idx"], dtype=np.int64),
                        "final_test": np.asarray(split["test_idx"], dtype=np.int64),
                    }
                sets = {role: set(map(int, values)) for role, values in arrays.items()}
                if any(len(values) != len(sets[role]) for role, values in arrays.items()):
                    raise AssertionError(f"duplicate split index: {source}")
                if (sets["train"] & sets["validation"] or
                        sets["train"] & sets["final_test"] or
                        sets["validation"] & sets["final_test"]):
                    raise AssertionError(f"overlapping split roles: {source}")
                paths, hashes = {}, {}
                for role, values in arrays.items():
                    relative = Path("split_roles") / f"{subject}_{paradigm}_fold{fold}_{role}.json"
                    path = output_root / relative
                    payload = canonical_bytes({"indices": values.tolist(), "role": role})
                    hashes[role] = write_new(path, payload)
                    paths[role] = relative.as_posix()
                records[(subject, paradigm, fold)] = {
                    "files": paths,
                    "sha256": hashes,
                    "counts": {role: len(values) for role, values in arrays.items()},
                }
    if len(records) != 300:
        raise AssertionError(f"expected 300 split cells, got {len(records)}")
    return records


def generate(run_root: Path, roles: dict[tuple[str, str, int], dict]) -> tuple[list, list, list]:
    stage1, stage2, stage3 = [], [], []
    for seed in SEEDS:
        for outer_fold in FOLDS:
            targets = TARGET_FOLDS[outer_fold]
            upstream = [subject for subject in SUBJECTS if subject not in targets]
            stage1.append({
                "stage": 1, "seed": seed, "fold": outer_fold,
                "outer_fold": outer_fold, "epochs": 150,
                "target_subjects": list(targets), "upstream_subjects": upstream,
                "target_subject_overlap": 0,
                "out_dir": str(run_root / "stage1" / f"seed{seed}" / f"outer_fold{outer_fold}"),
            })
            for variant in VARIANTS:
                stage2.append({
                    "stage": 2, "variant": variant, "seed": seed,
                    "outer_fold": outer_fold, "fold": outer_fold,
                    "epochs": 150, "weak_weight": 0.3,
                    "target_subjects": list(targets),
                    "upstream_subjects": upstream, "target_subject_overlap": 0,
                    "stage1_ckpt": (
                        str(run_root / "stage1" / f"seed{seed}" / f"outer_fold{outer_fold}" / "best_stage1.pth")
                        if variant == "full" else None
                    ),
                    "emg_access": "paired_teacher" if variant == "full" else "prohibited",
                    "out_dir": str(run_root / "stage2" / variant / f"seed{seed}" / f"outer_fold{outer_fold}"),
                })
    target_to_outer = {
        subject: outer_fold for outer_fold, targets in TARGET_FOLDS.items() for subject in targets
    }
    if set(target_to_outer) != set(SUBJECTS):
        raise AssertionError("outer target map must partition all 30 subjects")
    for variant in VARIANTS:
        for seed in SEEDS:
            for subject in SUBJECTS:
                outer_fold = target_to_outer[subject]
                for paradigm in PARADIGMS:
                    for inner_fold in FOLDS:
                        role = roles[(subject, paradigm, inner_fold)]
                        stage3.append({
                            "stage": 3, "variant": variant, "seed": seed,
                            "subject": subject, "outer_fold": outer_fold,
                            "fold": inner_fold, "epochs": 100,
                            "paradigm": paradigm, "inner_fold": inner_fold,
                            "stage2_ckpt": str(
                                run_root / "stage2" / variant / f"seed{seed}" /
                                f"outer_fold{outer_fold}" / "best_stage2.pth"
                            ),
                            "split_role_files": role["files"],
                            "split_role_sha256": role["sha256"],
                            "checkpoint_selection": "target_validation_balanced_accuracy_only",
                            "final_test_policy": "open_once_after_selected_checkpoint_reload",
                            "stage3_modality": "EEG_only",
                            "evaluate_zero_shot": True,
                            "encoder_lr": 1e-6, "classifier_lr": 1e-5,
                            "out_dir": str(
                                run_root / "stage3" / variant / f"seed{seed}" /
                                subject / paradigm / f"inner_fold{inner_fold}"
                            ),
                        })
    if (len(stage1), len(stage2), len(stage3)) != (15, 30, 1800):
        raise AssertionError("subject-isolated manifest count mismatch")
    return stage1, stage2, stage3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-split-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    roles = export_split_roles(args.source_split_root.resolve(), output)
    stage1, stage2, stage3 = generate(args.run_root.resolve(), roles)
    hashes = {
        "stage1_subject_isolated_15.jsonl": write_jsonl(output / "stage1_subject_isolated_15.jsonl", stage1),
        "stage2_subject_isolated_30.jsonl": write_jsonl(output / "stage2_subject_isolated_30.jsonl", stage2),
        "stage3_subject_isolated_1800.jsonl": write_jsonl(output / "stage3_subject_isolated_1800.jsonl", stage3),
    }
    audit = {
        "counts": {"stage1": 15, "stage2": 30, "stage3": 1800, "split_cells": 300},
        "outer_target_folds": {str(key): list(value) for key, value in TARGET_FOLDS.items()},
        "split_family": "outer_seed_20260811",
        "target_subject_overlap": 0,
        "sha256": hashes,
    }
    write_new(output / "manifest_audit.json", json.dumps(audit, indent=2, sort_keys=True).encode() + b"\n")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
