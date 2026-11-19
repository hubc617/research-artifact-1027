#!/usr/bin/env python3
"""Aggregate the seed-42 SmoothGrad task CSVs into the two Fig. 2 inputs."""
from __future__ import annotations

import argparse
import csv
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np

from generate_secondary_manifests import FOLDS, PARADIGMS, SUBJECTS, VARIANTS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from privileged_emg.secondary import EEG_CHANNELS, OUTER_RING_WIDE

KEY_FIELDS = ("paradigm", "variant", "subject", "fold", "seed")
CSV_NAME = "channel_attribution.csv"
SUM_TOLERANCE = 2e-5


def task_csvs(manifest):
    expected = set(product(PARADIGMS, VARIANTS, SUBJECTS, FOLDS, (42,)))
    tasks = {}
    paths = set()
    with manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            location = f"{manifest}:{line_number}"
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("analysis") != "smoothgrad_abs":
                raise ValueError(f"{location}: expected a SmoothGrad-Abs manifest row")
            key = tuple(row.get(field) for field in KEY_FIELDS)
            if (
                not all(type(value) is str for value in key[:3])
                or not all(type(value) is int for value in key[3:])
                or key not in expected
            ):
                raise ValueError(f"{location}: invalid task identity {key}")
            if key in tasks:
                raise ValueError(f"{location}: duplicate task identity {key}")
            output = row.get("output")
            if not isinstance(output, str) or not output.strip():
                raise ValueError(f"{location}: expected the task output directory")
            path = (Path(output) / CSV_NAME).resolve()
            if path in paths:
                raise ValueError(f"{location}: duplicate task CSV {path}")
            paths.add(path)
            tasks[key] = path
    if missing := expected - tasks.keys():
        raise ValueError(f"{manifest}: missing {len(missing)} task(s), including {min(missing)}")
    return tasks


def read_channel_shares(path, task):
    required = set(KEY_FIELDS) | {"channel", "relative_attribution"}
    values = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not required.issubset(fields) or len(set(fields)) != len(fields):
            raise ValueError(f"{path}: expected unique columns including {sorted(required)}")
        identity = tuple(str(value) for value in task)
        for line, row in enumerate(reader, start=2):
            if tuple(row[field] for field in KEY_FIELDS) != identity:
                raise ValueError(f"{path}:{line}: expected task identity {task}")
            channel = row["channel"]
            if channel not in EEG_CHANNELS or channel in values:
                raise ValueError(f"{path}:{line}: unknown or duplicate channel {channel!r}")
            try:
                share = float(row["relative_attribution"])
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{line}: invalid relative_attribution") from error
            if not 0.0 <= share <= 1.0:
                raise ValueError(f"{path}:{line}: relative_attribution must be a finite fraction")
            values[channel] = share
    if missing := set(EEG_CHANNELS) - values.keys():
        raise ValueError(f"{path}: missing channels {sorted(missing)}")
    vector = np.array([values[channel] for channel in EEG_CHANNELS], dtype=np.float64)
    total = vector.sum()
    if not np.isclose(total, 1.0, rtol=0.0, atol=SUM_TOLERANCE):
        raise ValueError(f"{path}: channel fractions sum to {total:.10g}, expected 1")
    return vector


def aggregate(tasks):
    shares = {task: read_channel_shares(path, task) for task, path in tasks.items()}
    outer = [EEG_CHANNELS.index(channel) for channel in OUTER_RING_WIDE]
    channels, annotations = [], []
    for paradigm in PARADIGMS:
        subject_shares = {
            variant: np.array([
                np.mean([shares[paradigm, variant, subject, fold, 42] for fold in FOLDS], axis=0)
                for subject in SUBJECTS
            ])
            for variant in VARIANTS
        }
        full, no_distill = subject_shares["full"], subject_shares["no_distill"]
        delta = full - no_distill
        channels.extend(zip(
            [paradigm] * len(EEG_CHANNELS), EEG_CHANNELS, delta.mean(axis=0), strict=True
        ))
        peripheral = full[:, outer].sum(axis=1) - no_distill[:, outer].sum(axis=1)
        annotations.append((
            paradigm, 100.0 * peripheral.mean(), np.count_nonzero(peripheral > 0), len(SUBJECTS)
        ))
    return channels, annotations


def write_csv(path, header, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        channels, annotations = aggregate(task_csvs(args.manifest))
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_csv(
            args.output_dir / "fig2_channel_summary.csv",
            ("paradigm", "channel", "full_minus_no_distill"),
            channels,
        )
        write_csv(
            args.output_dir / "fig2_annotation_summary.csv",
            ("paradigm", "shift_pp", "positive_subjects", "total_subjects"),
            annotations,
        )
    except (OSError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
