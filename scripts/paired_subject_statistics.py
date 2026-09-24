#!/usr/bin/env python3
"""Compute prespecified paired subject-level bootstrap and sign-flip statistics."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


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
    frame = pd.read_csv(args.input)
    required = {"subject", "paradigm", "variant", "balanced_accuracy"}
    if missing := required - set(frame):
        raise ValueError(f"missing columns: {sorted(missing)}")
    cell_columns = {"seed", "fold"}
    if cell_columns.issubset(frame.columns):
        keys = ["subject", "paradigm", "variant"]
        counts = frame.groupby(keys).size()
        if not (counts == 15).all():
            raise ValueError(
                "raw three-seed/five-fold input must contain exactly 15 cells "
                "per subject, paradigm, and variant"
            )
        if set(frame["seed"].astype(int)) != {42, 2026, 3407}:
            raise ValueError("expected seeds 42, 2026, and 3407")
        if set(frame["fold"].astype(int)) != set(range(5)):
            raise ValueError("expected folds 0-4")
        seed_level = frame.groupby([*keys, "seed"], as_index=False)["balanced_accuracy"].mean()
        frame = seed_level.groupby(keys, as_index=False)["balanced_accuracy"].mean()
        aggregation = "mean five folds within each seed, then mean three seeds"
        n_source_cells_per_subject = 15
    else:
        if frame.duplicated(["subject", "paradigm", "variant"]).any():
            raise ValueError("subject-level input contains duplicate cells")
        aggregation = "input already contains one estimate per subject/paradigm/variant"
        n_source_cells_per_subject = 1
    rows = []
    for offset, paradigm in enumerate(("overt", "silent_1001")):
        part = frame[frame.paradigm == paradigm].pivot(
            index="subject", columns="variant", values="balanced_accuracy"
        )
        if set(part.index) != {f"S{i:02d}" for i in range(1, 31)}:
            raise ValueError(f"{paradigm}: expected exactly S01-S30")
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
