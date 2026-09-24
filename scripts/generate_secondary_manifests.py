#!/usr/bin/env python3
"""Generate deterministic seed-42 channel-removal and SmoothGrad manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SUBJECTS = tuple(f"S{i:02d}" for i in range(1, 31))
PARADIGMS = ("overt", "silent_1001")
FOLDS = tuple(range(5))
VARIANTS = ("full", "no_distill")
CHANNELS = (
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "O1", "OZ", "O2",
)
DROPPED = (
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F6", "F8", "FT7",
    "FC5", "FC6", "FT8", "T7", "C5", "C6", "T8", "TP7", "CP5", "CP6",
    "TP8", "P7", "P5", "P6", "P8", "PO7", "PO5", "PO6", "PO8", "O1",
    "OZ", "O2",
)


def write(path: Path, rows: list[dict]) -> str:
    payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    channel, smoothgrad = [], []
    for variant in VARIANTS:
        for fold in FOLDS:
            stage2 = args.checkpoint_root / "stage2" / variant / "seed42" / f"fold{fold}" / "best_stage2.pth"
            for subject in SUBJECTS:
                for paradigm in PARADIGMS:
                    identity = {
                        "variant": variant, "seed": 42, "fold": fold,
                        "subject": subject, "paradigm": paradigm,
                    }
                    channel.append({
                        **identity, "analysis": "channel_removal_28_channel",
                        "stage2_checkpoint": str(stage2),
                        "input_channels": list(CHANNELS),
                        "zero_mask_channels": list(DROPPED),
                        "mask_application": ["train", "validation", "final_test"],
                        "stage2_unchanged": True,
                        "output": str(args.run_root / "channel_removal" / variant / f"fold{fold}" / subject / paradigm),
                    })
                    smoothgrad.append({
                        **identity, "analysis": "smoothgrad_abs",
                        "stage3_checkpoint": str(
                            args.run_root / "stage3" / variant / "seed42" / f"fold{fold}" /
                            subject / paradigm / "best_stage3.pth"
                        ),
                        "target": "true_class_logit", "noise_samples": 16,
                        "noise_sd": "0.1_times_within_trial_eeg_sd",
                        "operation_order": "abs_gradient_then_noise_mean_then_time_mean_then_trial_L1",
                        "output": str(args.run_root / "smoothgrad" / variant / f"fold{fold}" / subject / paradigm),
                    })
    if len(channel) != 600 or len(smoothgrad) != 600:
        raise AssertionError("secondary manifest count mismatch")
    hashes = {
        "channel_removal_seed42_600.jsonl": write(output / "channel_removal_seed42_600.jsonl", channel),
        "smoothgrad_seed42_600.jsonl": write(output / "smoothgrad_seed42_600.jsonl", smoothgrad),
    }
    metadata = {
        "counts": {"channel_removal": 600, "smoothgrad": 600},
        "channel_count": len(CHANNELS), "zero_mask_count": len(DROPPED),
        "retained_channel_count": len(CHANNELS) - len(DROPPED), "sha256": hashes,
    }
    (output / "manifest_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
