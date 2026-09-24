import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import yaml

from privileged_emg.model_MAE import MAE_Encoder
from privileged_emg.model_stage2 import (
    EEGMAEAlignerPretrainer,
    HierarchicalDistillLoss,
    NoDistillPretrainer,
)
from privileged_emg.secondary import EEG_CHANNELS, OUTER_RING_WIDE

ROOT = Path(__file__).resolve().parents[1]


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_shared_encoder_architecture_and_patch_sizes() -> None:
    eeg = MAE_Encoder(60, embed_dim=256, patch_size=16, seq_len=256)
    emg = MAE_Encoder(6, embed_dim=256, patch_size=25, seq_len=1000)
    layer = eeg.transformer.layers[0]
    assert len(eeg.transformer.layers) == 4
    assert layer.self_attn.num_heads == 8
    assert layer.linear1.out_features == 1024
    assert eeg.patch_size == 16 and emg.patch_size == 25


def test_class_prototype_empty_same_class_falls_back_to_instance() -> None:
    eeg = torch.eye(3, 4)
    emg = eeg.clone()
    labels = torch.tensor([0, 1, 2])
    loss = HierarchicalDistillLoss(weak_weight=0.3)(eeg, emg, labels)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_no_distill_trainable_initialization_matches_full(monkeypatch) -> None:
    monkeypatch.setattr(EEGMAEAlignerPretrainer, "_load_emg_encoder", lambda self, path: None)
    torch.manual_seed(42)
    full = EEGMAEAlignerPretrainer("not-opened")
    torch.manual_seed(42)
    baseline = NoDistillPretrainer()
    for prefix in ("eeg_encoder", "eeg_decoder", "eeg_proj", "classifier"):
        left = getattr(full, prefix).state_dict()
        right = getattr(baseline, prefix).state_dict()
        assert left.keys() == right.keys()
        assert all(torch.equal(left[key], right[key]) for key in left)


def test_stage3_search_contains_exact_eight_candidates() -> None:
    config = yaml.safe_load((ROOT / "configs/stage3_validation_search.yaml").read_text())
    assert tuple(config["candidates"]) == tuple("ABCDEFGH")
    assert config["frozen_winner"] == "E"
    assert config["candidates"]["E"] == {
        "encoder_lr": 3e-6, "classifier_lr": 1e-4,
        "dropout": 0.5, "label_smoothing": 0.1,
    }
    assert config["final_test_during_selection"] == "prohibited"


def test_submitted_results_baseline_and_current_readme() -> None:
    submitted = json.loads((ROOT / "docs/SUBMITTED_RESULTS_20260919.json").read_text())
    readme = (ROOT / "README.md").read_text()
    expected = {
        "Full (α=0.3)": 48.61,
        "NoDistill": 41.23,
        "Unrestricted shuffle": 41.23,
        "Within-class shuffle": 48.11,
        "Instance-only (α=0)": 48.74,
        "LabelSimplex": 38.40,
    }
    stage2 = {row["method"]: row["stage2_validation_ba"] for row in submitted["main_table"]}
    for label, value in expected.items():
        assert stage2[label] == value
        assert f"{value:.2f}" in readme
    assert "`53.26%` Overt and `40.12%` Silent" in readme
    assert "(`54.53/41.17`) is not substituted or relabeled" in readme


def test_fig2_v7_source_preserves_release_invariants() -> None:
    source = (ROOT / "scripts/make_fig2_v7.py").read_text()
    assert "pdf.fonttype" in source and "42" in source
    assert "contour" in source
    assert "argparse" in source
    assert "54.53" not in source and "41.17" not in source


def test_six_variant_and_clean_main_manifests_are_deterministic(tmp_path) -> None:
    script = ROOT / "scripts/generate_protocol_manifests.py"
    base = [
        sys.executable, str(script), "--data-root", str(tmp_path / "data"),
        "--split-root", str(tmp_path / "splits"),
        "--run-root", str(tmp_path / "runs"),
    ]
    first, second = tmp_path / "first", tmp_path / "second"
    subprocess.run([*base, "--output-dir", str(first)], check=True, capture_output=True)
    subprocess.run([*base, "--output-dir", str(second)], check=True, capture_output=True)
    expected = {
        "stage2_six_variant_90.jsonl": 90,
        "stage3_six_variant_5400.jsonl": 5400,
        "stage2_clean_main_four_variant_60.jsonl": 60,
        "stage3_clean_main_four_variant_3600.jsonl": 3600,
    }
    for name, count in expected.items():
        assert (first / name).read_bytes() == (second / name).read_bytes()
        assert len(lines(first / name)) == count
    stage2 = lines(first / "stage2_six_variant_90.jsonl")
    for row in stage2:
        if row["variant"] in {"no_distill", "label_simplex"}:
            assert row["stage1_ckpt"] is None


def _write_synthetic_splits(root: Path) -> None:
    root.mkdir()
    train = np.arange(0, 6)
    validation = np.arange(6, 8)
    test = np.arange(8, 10)
    for subject in (f"S{i:02d}" for i in range(1, 31)):
        for paradigm in ("overt", "silent_1001"):
            for fold in range(5):
                np.savez(
                    root / f"{subject}_{paradigm}_fold{fold}.npz",
                    train_idx=train, val_idx=validation, test_idx=test,
                )


def test_subject_isolated_grid_excludes_all_outer_targets(tmp_path) -> None:
    split_root = tmp_path / "splits"
    _write_synthetic_splits(split_root)
    script = ROOT / "scripts/generate_subject_isolated_manifests.py"
    outputs = []
    for name in ("one", "two"):
        output = tmp_path / name
        subprocess.run([
            sys.executable, str(script), "--source-split-root", str(split_root),
            "--run-root", str(tmp_path / "runs"), "--output-dir", str(output),
        ], check=True, capture_output=True)
        outputs.append(output)
    for name, count in (
        ("stage1_subject_isolated_15.jsonl", 15),
        ("stage2_subject_isolated_30.jsonl", 30),
        ("stage3_subject_isolated_1800.jsonl", 1800),
    ):
        assert (outputs[0] / name).read_bytes() == (outputs[1] / name).read_bytes()
        assert len(lines(outputs[0] / name)) == count
    for row in lines(outputs[0] / "stage2_subject_isolated_30.jsonl"):
        assert not set(row["target_subjects"]) & set(row["upstream_subjects"])
        assert len(row["target_subjects"]) == 6
        assert len(row["upstream_subjects"]) == 24
    target_map = {
        (item["seed"], item["outer_fold"]): item["target_subjects"]
        for item in lines(outputs[0] / "stage1_subject_isolated_15.jsonl")
    }
    for row in lines(outputs[0] / "stage3_subject_isolated_1800.jsonl"):
        assert row["subject"] in target_map[(row["seed"], row["outer_fold"])]
        assert row["evaluate_zero_shot"] is True


def test_secondary_manifest_grid_and_mask(tmp_path) -> None:
    output = tmp_path / "secondary"
    subprocess.run([
        sys.executable, str(ROOT / "scripts/generate_secondary_manifests.py"),
        "--checkpoint-root", str(tmp_path / "checkpoints"),
        "--run-root", str(tmp_path / "runs"), "--output-dir", str(output),
    ], check=True, capture_output=True)
    channel = lines(output / "channel_removal_seed42_600.jsonl")
    smoothgrad = lines(output / "smoothgrad_seed42_600.jsonl")
    assert len(channel) == len(smoothgrad) == 600
    assert len(EEG_CHANNELS) == 60 and len(OUTER_RING_WIDE) == 32
    assert len(set(EEG_CHANNELS) - set(OUTER_RING_WIDE)) == 28
    assert all(row["mask_application"] == ["train", "validation", "final_test"] for row in channel)
    assert all(row["operation_order"].startswith("abs_gradient") for row in smoothgrad)
