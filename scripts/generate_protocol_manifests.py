#!/usr/bin/env python3
"""Generate deterministic source-only task manifests for the frozen protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SEEDS = (42, 2026, 3407)
FOLDS = range(5)
SUBJECTS = tuple(f"S{i:02d}" for i in range(1, 31))
PARADIGMS = ("overt", "silent_1001")
VARIANTS = (
    "full",
    "no_distill",
    "unrestricted_shuffle",
    "within_class_shuffle",
    "instance_only_alpha0",
)
LABEL_SIMPLEX = "label_simplex"
CLEAN_MAIN_VARIANTS = VARIANTS[:4]
SIX_VARIANTS = (*VARIANTS, LABEL_SIMPLEX)


def write_jsonl(path: Path, rows: list[dict]) -> str:
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)

    stage1 = [
        {"stage": 1, "seed": seed, "fold": fold, "epochs": 150,
         "out_dir": str(args.run_root / "stage1" / f"seed{seed}" / f"fold{fold}")}
        for seed in SEEDS for fold in FOLDS
    ]
    stage2 = [
        {"stage": 2, "variant": variant, "seed": seed, "fold": fold,
         "epochs": 150, "warmup_epochs": 30,
         "weak_weight": 0.0 if variant == "instance_only_alpha0" else 0.3,
         "stage1_ckpt": (
             None if variant == "no_distill" else
             str(args.run_root / "stage1" / f"seed{seed}" / f"fold{fold}" / "best_stage1.pth")
         ),
         "emg_access": "prohibited" if variant == "no_distill" else "paired_teacher",
         "out_dir": str(args.run_root / "stage2" / variant / f"seed{seed}" / f"fold{fold}")}
        for variant in VARIANTS for seed in SEEDS for fold in FOLDS
    ]
    stage3 = [
        {"stage": 3, "variant": variant, "seed": seed, "fold": fold,
         "subject": subject, "paradigm": paradigm, "epochs": 100,
         "stage3_modality": "EEG_only",
         "split": str(args.split_root / f"{subject}_{paradigm}_fold{fold}.npz"),
         "stage2_ckpt": str(args.run_root / "stage2" / variant / f"seed{seed}" / f"fold{fold}" / "best_stage2.pth"),
         "out_dir": str(args.run_root / "stage3" / variant / f"seed{seed}" / f"fold{fold}" / subject / paradigm)}
        for variant in VARIANTS for seed in SEEDS for fold in FOLDS
        for subject in SUBJECTS for paradigm in PARADIGMS
    ]
    label_simplex_stage2 = [
        {"stage": 2, "variant": LABEL_SIMPLEX, "seed": seed, "fold": fold,
         "epochs": 150, "warmup_epochs": 30, "stage1_ckpt": None,
         "emg_access": "prohibited",
         "out_dir": str(args.run_root / "stage2" / LABEL_SIMPLEX / f"seed{seed}" / f"fold{fold}")}
        for seed in SEEDS for fold in FOLDS
    ]
    label_simplex_stage3 = [
        {"stage": 3, "variant": LABEL_SIMPLEX, "seed": seed, "fold": fold,
         "subject": subject, "paradigm": paradigm, "epochs": 100,
         "stage3_modality": "EEG_only",
         "split": str(args.split_root / f"{subject}_{paradigm}_fold{fold}.npz"),
         "stage2_ckpt": str(args.run_root / "stage2" / LABEL_SIMPLEX / f"seed{seed}" / f"fold{fold}" / "best_stage2.pth"),
         "out_dir": str(args.run_root / "stage3" / LABEL_SIMPLEX / f"seed{seed}" / f"fold{fold}" / subject / paradigm)}
        for seed in SEEDS for fold in FOLDS
        for subject in SUBJECTS for paradigm in PARADIGMS
    ]
    stage2_six = [*stage2, *label_simplex_stage2]
    stage3_six = [*stage3, *label_simplex_stage3]
    stage2_clean_main = [row for row in stage2 if row["variant"] in CLEAN_MAIN_VARIANTS]
    stage3_clean_main = [row for row in stage3 if row["variant"] in CLEAN_MAIN_VARIANTS]
    expected = {
        "stage1": 15,
        "stage2_primary": 75,
        "stage3_primary": 4500,
        "label_simplex_stage2": 15,
        "label_simplex_stage3": 900,
        "stage2_six_variant": 90,
        "stage3_six_variant": 5400,
        "stage2_clean_main_four_variant": 60,
        "stage3_clean_main_four_variant": 3600,
    }
    observed = {
        "stage1": len(stage1),
        "stage2_primary": len(stage2),
        "stage3_primary": len(stage3),
        "label_simplex_stage2": len(label_simplex_stage2),
        "label_simplex_stage3": len(label_simplex_stage3),
        "stage2_six_variant": len(stage2_six),
        "stage3_six_variant": len(stage3_six),
        "stage2_clean_main_four_variant": len(stage2_clean_main),
        "stage3_clean_main_four_variant": len(stage3_clean_main),
    }
    if observed != expected:
        raise AssertionError(f"manifest grid mismatch: {observed} != {expected}")
    hashes = {
        "stage1_15.jsonl": write_jsonl(output / "stage1_15.jsonl", stage1),
        "stage2_75.jsonl": write_jsonl(output / "stage2_75.jsonl", stage2),
        "stage3_4500.jsonl": write_jsonl(output / "stage3_4500.jsonl", stage3),
        "label_simplex_stage2_15.jsonl": write_jsonl(
            output / "label_simplex_stage2_15.jsonl", label_simplex_stage2
        ),
        "label_simplex_stage3_900.jsonl": write_jsonl(
            output / "label_simplex_stage3_900.jsonl", label_simplex_stage3
        ),
        "stage2_six_variant_90.jsonl": write_jsonl(
            output / "stage2_six_variant_90.jsonl", stage2_six
        ),
        "stage3_six_variant_5400.jsonl": write_jsonl(
            output / "stage3_six_variant_5400.jsonl", stage3_six
        ),
        "stage2_clean_main_four_variant_60.jsonl": write_jsonl(
            output / "stage2_clean_main_four_variant_60.jsonl", stage2_clean_main
        ),
        "stage3_clean_main_four_variant_3600.jsonl": write_jsonl(
            output / "stage3_clean_main_four_variant_3600.jsonl", stage3_clean_main
        ),
    }
    metadata = {
        "data_root": str(args.data_root.resolve()), "split_root": str(args.split_root.resolve()),
        "run_root": str(args.run_root.resolve()),
        "counts": expected,
        "compatibility": {
            "stage2_75.jsonl": "legacy five-condition grid retained",
            "stage3_4500.jsonl": "legacy five-condition grid retained",
            "label_simplex_stage2_15.jsonl": "legacy separate control grid retained",
            "label_simplex_stage3_900.jsonl": "legacy separate control grid retained",
        },
        "variant_order": list(SIX_VARIANTS),
        "sha256": hashes,
    }
    (output / "manifest_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
