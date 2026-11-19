#!/usr/bin/env python3
"""Run one seed-42 Fig. 2 SmoothGrad-Abs task and write channel fractions."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from generate_secondary_manifests import (
    CHANNELS, FIG2_FIRST_NOISE_SEED, FIG2_NOISE_FRACTION, FIG2_PROTOCOL, FIG2_TASKS,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from privileged_emg.common import read_jsonl, set_seed
from privileged_emg.label_simplex import EEGIndexedDataset, stage3_data_paths
from privileged_emg.model_stage3 import EEGDownstreamClassifier

KEY_FIELDS = ("paradigm", "variant", "subject", "fold", "seed")
TASK_IDS = {key: index for index, key in enumerate(FIG2_TASKS)}


def validate_task(task: dict) -> int:
    if not isinstance(task, dict):
        raise ValueError("expected a JSON object for the selected Fig. 2 task")
    key = tuple(task.get(field) for field in ("variant", "fold", "subject", "paradigm"))
    if (
        not all(isinstance(key[i], str) for i in (0, 2, 3))
        or type(key[1]) is not int or key not in TASK_IDS
        or type(task.get("seed")) is not int or task["seed"] != 42
    ):
        raise ValueError(f"invalid seed-42 Fig. 2 task identity {key}")
    for field, expected in FIG2_PROTOCOL.items():
        if task.get(field) != expected:
            raise ValueError(f"{key}: expected {field}={expected!r}; regenerate the Fig. 2 manifest")
    expected_seed = FIG2_FIRST_NOISE_SEED + TASK_IDS[key]
    if type(task.get("noise_seed")) is not int or task["noise_seed"] != expected_seed:
        raise ValueError(f"{key}: expected noise_seed={expected_seed}")
    return expected_seed


def task_path(task: dict, field: str, override: Path | None) -> Path:
    if override is not None:
        return override
    value = task.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest is missing a path for {field}")
    return Path(value)


def read_final_indices(split_path: Path) -> np.ndarray:
    with np.load(split_path, allow_pickle=False) as split:
        if "test_idx" not in split:
            raise ValueError(f"{split_path}: missing test_idx")
        indices = split["test_idx"]
    if (
        indices.ndim != 1 or not indices.size or indices.dtype.kind not in "iu"
        or np.unique(indices).size != indices.size
    ):
        raise ValueError(f"{split_path}: expected nonempty, unique integer test_idx")
    return indices


def noisy_trials(trial: torch.Tensor) -> torch.Tensor:
    sigma = FIG2_NOISE_FRACTION * float(trial.std(unbiased=False).item())
    noise = torch.randn(
        (FIG2_PROTOCOL["noise_samples"], *trial.shape),
        device=trial.device, dtype=trial.dtype,
    )
    return trial.unsqueeze(0) + sigma * noise


def channel_attribution(model, dataset: EEGIndexedDataset, device: torch.device) -> np.ndarray:
    model.eval()
    model.requires_grad_(False)
    # The archived prediction replay consumed RNG in the encoder even with
    # mask_ratio=0. Keep the same forward batches and order before drawing noise.
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
    with torch.no_grad():
        for eeg, _ in loader:
            model(eeg.to(device), labels=None, mask_ratio=0.0)

    total = np.zeros((60, 256), dtype=np.float64)
    with torch.enable_grad():
        for trial, label in dataset:
            noisy = noisy_trials(trial.to(device)).requires_grad_(True)
            logits = model(noisy, labels=None, mask_ratio=0.0)["logits"]
            gradient, = torch.autograd.grad(logits[:, int(label)].sum(), noisy)
            attribution = gradient.abs().mean(0)
            norm = attribution.sum()
            if not torch.isfinite(norm) or norm <= 0:
                raise ValueError("expected finite, nonzero trial attribution")
            relative = (attribution / norm).detach().cpu().numpy()
            if abs(float(relative.sum()) - 1.0) > 2e-5:
                raise ValueError("trial attribution fractions must sum to one")
            total += relative

    mean = (total / len(dataset)).astype(np.float32)
    channel = mean.sum(axis=1, dtype=np.float32)
    if (
        channel.shape != (len(CHANNELS),) or not np.isfinite(channel).all()
        or (channel < 0).any() or (channel > 1).any()
        or abs(float(channel.sum()) - 1.0) > 2e-5
    ):
        raise ValueError("expected 60 finite, nonnegative channel fractions summing to one")
    return channel


def run(args: argparse.Namespace) -> Path:
    task = read_jsonl(args.manifest, args.index)
    noise_seed = validate_task(task)
    output = task_path(task, "output", args.output_dir)
    if output.exists():
        raise FileExistsError(f"expected a new task output directory: {output}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for a functional smoke test")

    eeg_path, label_path = stage3_data_paths(args.data_root, task["subject"], task["paradigm"])
    split_path = args.split_root / f"{task['subject']}_{task['paradigm']}_fold{task['fold']}.npz"
    indices = read_final_indices(split_path)
    dataset = EEGIndexedDataset(eeg_path, label_path, indices, "final_test")
    if dataset.labels.ndim != 1 or not torch.isfinite(dataset.eeg).all():
        raise ValueError("expected one label per trial and finite final-test EEG")

    set_seed(noise_seed)
    stage2 = task_path(task, "stage2_checkpoint", args.stage2_checkpoint)
    stage3 = task_path(task, "stage3_checkpoint", args.stage3_checkpoint)
    model = EEGDownstreamClassifier(stage2, 10).to(device)
    state = torch.load(stage3, map_location=device, weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    fractions = channel_attribution(model, dataset, device)

    output.mkdir(parents=True, exist_ok=False)
    path = output / "channel_attribution.csv"
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow((*KEY_FIELDS, "channel", "relative_attribution"))
        identity = tuple(task[field] for field in KEY_FIELDS)
        for channel, fraction in zip(CHANNELS, fractions, strict=True):
            writer.writerow((*identity, channel, float(fraction)))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True, help="Zero-based JSONL row to run")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, help="Override the selected row's Stage 2 path")
    parser.add_argument("--stage3-checkpoint", type=Path, help="Override the selected row's Stage 3 path")
    parser.add_argument("--output-dir", type=Path, help="Override the selected row's new output directory")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.index < 0:
        parser.error("--index must be zero or greater")
    try:
        print(run(args))
    except (AssertionError, IndexError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")


if __name__ == "__main__":
    main()
