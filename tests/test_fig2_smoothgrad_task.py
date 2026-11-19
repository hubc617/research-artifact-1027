import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from privileged_emg.model_MAE import MAE_Encoder
from privileged_emg.model_stage3 import EEGDownstreamClassifier


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def test_noise_scale_is_population_std_of_the_entire_trial(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from run_fig2_smoothgrad_task import noisy_trials

    scales = torch.linspace(0.5, 6.0, 60).unsqueeze(1)
    trial = scales * torch.linspace(-1.0, 1.0, 256)
    trial[0] += 3.0

    def unit_noise(shape, *, device, dtype):
        assert shape == (16, 60, 256)
        assert device == trial.device and dtype == trial.dtype
        return torch.ones(shape, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randn", unit_noise)
    expected_sigma = 0.1 * np.std(trial.numpy().astype(np.float64))
    torch.testing.assert_close(noisy_trials(trial), (trial + expected_sigma).expand(16, -1, -1))
    assert not np.isclose(expected_sigma, 0.1 * trial[0].std().item())


def test_gradient_draws_follow_prediction_replay(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from run_fig2_smoothgrad_task import channel_attribution, noisy_trials
    from torch.utils.data import DataLoader, TensorDataset

    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def forward(self, eeg, labels, mask_ratio):
            assert labels is None and mask_ratio == 0.0 and not self.training
            self.inputs.append((torch.is_grad_enabled(), eeg.detach().clone()))
            torch.rand(len(eeg), 16)
            return {"logits": eeg.square().sum((1, 2), keepdim=False).unsqueeze(1)}

    eeg = torch.stack((torch.ones(60, 256), torch.arange(60 * 256).reshape(60, 256) / 1000))
    labels = torch.zeros(len(eeg), dtype=torch.int64)
    dataset = TensorDataset(eeg, labels)
    model = RecordingModel()
    torch.manual_seed(100042)
    channel_attribution(model, dataset, torch.device("cpu"))
    assert [grad_enabled for grad_enabled, _ in model.inputs] == [False, True, True]
    assert torch.equal(model.inputs[0][1], eeg)

    torch.manual_seed(100042)
    # DataLoader construction of an iterator also advances the CPU RNG.
    replay_eeg, _ = next(iter(DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)))
    torch.rand(len(replay_eeg), 16)
    for trial, (_, actual) in zip(eeg, model.inputs[1:], strict=True):
        assert torch.equal(actual, noisy_trials(trial))
        torch.rand(len(actual), 16)


def cli(script, *args, check=True):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *map(str, args)],
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        capture_output=True, text=True, check=check,
    )


def test_fig2_cli_writes_repeatable_aggregator_input(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    from aggregate_smoothgrad import read_channel_shares
    from run_fig2_smoothgrad_task import validate_task

    source, splits, manifests = (tmp_path / name for name in ("data", "splits", "manifests"))
    runs = tmp_path / "runs"
    cli(
        "generate_secondary_manifests.py", "--checkpoint-root", runs, "--run-root", runs,
        "--output-dir", manifests,
    )
    manifest = manifests / "smoothgrad_seed42_600.jsonl"
    tasks = [json.loads(line) for line in manifest.read_text().splitlines()]
    task = tasks[0]
    assert validate_task(task) == 100042
    assert validate_task(tasks[300]) == 100342
    with pytest.raises(ValueError, match="noise_seed"):
        validate_task({**task, "noise_seed": 100342})

    participant = source / "overt" / "S01"
    participant.mkdir(parents=True)
    x = np.random.default_rng(1).normal(size=(3, 60, 256)).astype(np.float32)
    np.save(participant / "S01_eeg_feat.npy", x)
    np.save(participant / "S01_labels.npy", np.array([1, 3, 8]))
    splits.mkdir()
    np.savez(splits / "S01_overt_fold0.npz", test_idx=np.array([2, 0]))

    torch.manual_seed(7)
    stage2, stage3 = (Path(task[key]) for key in ("stage2_checkpoint", "stage3_checkpoint"))
    stage2.parent.mkdir(parents=True)
    stage3.parent.mkdir(parents=True)
    encoder = MAE_Encoder(60, embed_dim=256, patch_size=16, seq_len=256)
    torch.save({f"eeg_encoder.{k}": v for k, v in encoder.state_dict().items()}, stage2)
    torch.save(EEGDownstreamClassifier(stage2).state_dict(), stage3)

    command = (
        "--data-root", source, "--split-root", splits, "--device", "cpu",
    )
    cli("run_fig2_smoothgrad_task.py", *command, "--manifest", manifest, "--index", 0)
    first = Path(task["output"]) / "channel_attribution.csv"
    vector = read_channel_shares(first, ("overt", "full", "S01", 0, 42))
    assert vector.shape == (60,) and vector.sum() == pytest.approx(1, abs=2e-5)

    reordered = tmp_path / "reordered.jsonl"
    reordered.write_text("\n".join(map(json.dumps, [tasks[300], task])) + "\n")
    second_dir = tmp_path / "second"
    cli(
        "run_fig2_smoothgrad_task.py", *command, "--manifest", reordered, "--index", 1,
        "--output-dir", second_dir,
    )
    assert first.read_bytes() == (second_dir / first.name).read_bytes()
    failed = cli(
        "run_fig2_smoothgrad_task.py", *command, "--manifest", manifest, "--index", 0,
        check=False,
    )
    assert failed.returncode != 0
    assert "expected a new task output directory" in failed.stderr
