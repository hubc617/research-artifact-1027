#!/usr/bin/env python3
"""Compute prespecified paired subject-level bootstrap and sign-flip statistics."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PARADIGMS = ("overt", "silent_1001")
SUBJECTS = {f"S{i:02d}" for i in range(1, 31)}
CELLS = {(seed, fold) for seed in (42, 2026, 3407) for fold in range(5)}
GROUP_KEYS = ["subject", "paradigm", "variant"]


def validate_pairs(frame: pd.DataFrame, left: str, right: str) -> tuple[pd.DataFrame, bool]:
    required = {*GROUP_KEYS, "balanced_accuracy"}
    if missing := required - set(frame):
        raise ValueError(f"missing columns: {sorted(missing)}")
    if left == right:
        raise ValueError("--left and --right must select distinct conditions")

    task_columns = {"seed", "fold"} & set(frame)
    if task_columns and len(task_columns) != 2:
        raise ValueError("task-level inputs must supply both seed and fold")

    selected = frame.loc[frame.variant.isin((left, right))].copy()
    if not selected.paradigm.isin(PARADIGMS).all():
        values = selected.loc[~selected.paradigm.isin(PARADIGMS), "paradigm"].unique()
        raise ValueError(f"unexpected paradigm for selected conditions: {values.tolist()}")

    for paradigm in PARADIGMS:
        for variant in (left, right):
            subjects = set(selected.loc[
                (selected.paradigm == paradigm) & (selected.variant == variant), "subject"
            ])
            if subjects != SUBJECTS:
                raise ValueError(
                    f"{paradigm}/{variant}: expected S01-S30, "
                    f"missing={sorted(SUBJECTS - subjects)}, "
                    f"unexpected={sorted(map(str, subjects - SUBJECTS))}"
                )

    for (subject, paradigm, variant), group in selected.groupby(GROUP_KEYS):
        identity = f"{subject}/{paradigm}/{variant}"
        accuracy = pd.to_numeric(group.balanced_accuracy, errors="coerce")
        if not np.isfinite(accuracy).all() or not accuracy.between(0, 100).all():
            raise ValueError(f"{identity}: balanced_accuracy must be finite and in 0–100 percent")
        if not task_columns:
            if len(group) != 1:
                raise ValueError(f"{identity}: subject-level input must contain exactly one row")
            continue

        seed = pd.to_numeric(group.seed, errors="coerce")
        fold = pd.to_numeric(group.fold, errors="coerce")
        if not (seed.isin([42, 2026, 3407]).all() and fold.isin(range(5)).all()):
            raise ValueError(f"{identity}: expected integer seed in (42, 2026, 3407) and fold in 0–4")
        cells = list(zip(seed, fold))
        if len(cells) != len(set(cells)) or set(cells) != CELLS:
            raise ValueError(
                f"{identity}: expected each seed/fold once, "
                f"duplicates={len(cells) - len(set(cells))}, missing={sorted(CELLS - set(cells))}"
            )

    selected["balanced_accuracy"] = pd.to_numeric(selected.balanced_accuracy)
    if task_columns:
        selected = selected.astype({"seed": int, "fold": int})
    return selected, bool(task_columns)


def sign_flip(values: np.ndarray, draws: int, rng: np.random.Generator) -> float:
    observed = abs(float(values.mean()))
    exceed = 0
    for start in range(0, draws, 20_000):
        size = min(20_000, draws - start)
        means = (rng.choice((-1.0, 1.0), (size, len(values))) * values).mean(axis=1)
        exceed += int(np.count_nonzero(np.abs(means) >= observed - 1e-12))
    return float((exceed + 1) / (draws + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=50_000)
    parser.add_argument("--sign-flip-draws", type=int, default=200_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260917)
    parser.add_argument("--sign-flip-seed", type=int, default=20260918)
    args = parser.parse_args()
    frame, task_level = validate_pairs(pd.read_csv(args.input), args.left, args.right)
    if task_level:
        seed_level = frame.groupby([*GROUP_KEYS, "seed"], as_index=False)["balanced_accuracy"].mean()
        frame = seed_level.groupby(GROUP_KEYS, as_index=False)["balanced_accuracy"].mean()
        aggregation = "mean five folds within each seed, then mean three seeds"
        n_source_cells_per_subject = 15
    else:
        aggregation = "input already contains one estimate per subject/paradigm/variant"
        n_source_cells_per_subject = 1
    rows = []
    for offset, paradigm in enumerate(PARADIGMS):
        part = frame[frame.paradigm == paradigm].pivot(
            index="subject", columns="variant", values="balanced_accuracy"
        )
        values = (part[args.left] - part[args.right]).to_numpy(float)
        bootstrap_seed = args.bootstrap_seed + offset
        sign_flip_seed = args.sign_flip_seed + offset
        bootstrap_rng = np.random.default_rng(bootstrap_seed)
        sign_flip_rng = np.random.default_rng(sign_flip_seed)
        index = bootstrap_rng.integers(0, len(values), (args.bootstrap_draws, len(values)))
        boot = values[index].mean(axis=1)
        rows.append({
            "paradigm": paradigm, "comparison": f"{args.left} - {args.right}",
            "n_subjects": len(values), "mean_difference_pp": float(values.mean()),
            "unit": "percentage_points", "aggregation": aggregation,
            "n_source_cells_per_subject": n_source_cells_per_subject,
            "bootstrap_draws": args.bootstrap_draws,
            "bootstrap_rng_seed": bootstrap_seed,
            "bootstrap_95_ci_low_pp": float(np.quantile(boot, 0.025)),
            "bootstrap_95_ci_high_pp": float(np.quantile(boot, 0.975)),
            "sign_flip_draws": args.sign_flip_draws,
            "sign_flip_rng_seed": sign_flip_seed,
            "sign_flip_plus_one_correction": True,
            "sign_flip_p_two_sided": sign_flip(values, args.sign_flip_draws, sign_flip_rng),
            "positive_subjects": int((values > 0).sum()),
            "negative_subjects": int((values < 0).sum()),
            "zero_subjects": int((values == 0).sum()),
        })
    if args.output.exists():
        raise FileExistsError(args.output)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
