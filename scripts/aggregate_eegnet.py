"""Strictly audit and aggregate EEGNet results using the paper's hierarchy."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


METRICS = ("test_balanced_accuracy", "test_macro_f1", "test_accuracy")


def read_manifest(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_rows(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, members in sorted(groups.items()):
        result = dict(zip(group_keys, key))
        result["n_cells"] = len(members)
        for metric in METRICS:
            result[metric] = float(np.mean([float(member[metric]) for member in members]))
        output.append(result)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    tasks = read_manifest(args.manifest)
    rows, missing, invalid = [], [], []
    for index, task in enumerate(tasks):
        out = Path(task["out_dir"])
        result_path = out / "result.json"
        prediction_path = out / "predictions.npz"
        checkpoint_path = out / "best_eegnet.pth"
        if not (result_path.is_file() and prediction_path.is_file() and checkpoint_path.is_file()):
            missing.append(index)
            continue
        try:
            result = json.loads(result_path.read_text())
            predictions = np.load(prediction_path)
            split = np.load(task["split_file"])
            if result.get("status") != "complete":
                raise ValueError("status is not complete")
            if len(predictions["y_true"]) != int(result["n_test"]):
                raise ValueError("prediction count does not match n_test")
            if not np.array_equal(
                np.sort(predictions["trial_indices"]), np.sort(split["test_idx"])
            ):
                raise ValueError("prediction trial indices do not match test split")
            for key in METRICS:
                if not np.isfinite(float(result[key])):
                    raise ValueError(f"non-finite metric: {key}")
            rows.append(result)
        except Exception as error:
            invalid.append({"index": index, "error": repr(error), "out_dir": str(out)})

    print(f"EXPECTED=900 VALID={len(rows)} MISSING={len(missing)} INVALID={len(invalid)}")
    if missing:
        print("FIRST_MISSING_INDICES=", missing[:20])
    if invalid:
        print("FIRST_INVALID=", invalid[:5])
    if (missing or invalid) and not args.allow_incomplete:
        raise SystemExit("Strict aggregation stopped because the run is incomplete or invalid")
    if not rows:
        return

    combinations = Counter(
        (int(row["seed"]), int(row["fold"]), row["subject"], row["paradigm"])
        for row in rows
    )
    duplicates = [key for key, count in combinations.items() if count != 1]
    if duplicates:
        raise RuntimeError(f"Duplicate task identities: {duplicates[:5]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = sorted(
        rows,
        key=lambda row: (int(row["seed"]), int(row["fold"]), row["subject"], row["paradigm"]),
    )
    write_csv(args.out_dir / "eegnet_all_tasks.csv", all_rows)

    # First average five folds within each subject and seed, exactly matching
    # the primary paper aggregation hierarchy.
    subject_seed = mean_rows(all_rows, ("seed", "subject", "paradigm"))
    if not args.allow_incomplete and any(int(row["n_cells"]) != 5 for row in subject_seed):
        raise RuntimeError("Not every subject/seed/paradigm cell contains five folds")
    write_csv(args.out_dir / "eegnet_subject_seed_5fold_mean.csv", subject_seed)

    # Then average the three seed-level values within each subject.
    subject_final = mean_rows(subject_seed, ("subject", "paradigm"))
    if not args.allow_incomplete and any(int(row["n_cells"]) != 3 for row in subject_final):
        raise RuntimeError("Not every subject/paradigm cell contains three seeds")
    write_csv(args.out_dir / "eegnet_subject_5fold_3seed_mean.csv", subject_final)

    rng = np.random.default_rng(20260914)
    summary = []
    for paradigm in ("overt", "silent_1001"):
        members = [row for row in subject_final if row["paradigm"] == paradigm]
        if not args.allow_incomplete and len(members) != 30:
            raise RuntimeError(f"{paradigm}: expected 30 subjects, got {len(members)}")
        row = {"baseline": "EEGNet-8,2", "paradigm": paradigm, "n_subjects": len(members)}
        for metric in METRICS:
            values = np.asarray([float(member[metric]) for member in members])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_median"] = float(np.median(values))
            if metric == "test_balanced_accuracy":
                draw = rng.integers(0, len(values), size=(50000, len(values)))
                boot = values[draw].mean(axis=1)
                row["test_balanced_accuracy_ci95_low"] = float(np.percentile(boot, 2.5))
                row["test_balanced_accuracy_ci95_high"] = float(np.percentile(boot, 97.5))
        summary.append(row)
    write_csv(args.out_dir / "eegnet_final_summary.csv", summary)
    (args.out_dir / "eegnet_final_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not missing and not invalid:
        print("EEGNET_STRICT_900_TASK_AUDIT_OK")


if __name__ == "__main__":
    main()
