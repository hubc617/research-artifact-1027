from pathlib import Path
import re

import yaml

from privileged_emg.cli import LEGACY_VARIANTS, VARIANTS
from privileged_emg.eegnet import EEGNet82, architecture_dict
from privileged_emg.model_stage2 import HierarchicalDistillLoss


ROOT = Path(__file__).resolve().parents[1]


def test_instance_only_is_alpha_zero() -> None:
    config = yaml.safe_load((ROOT / "configs/stage2_instance_only_alpha0.yaml").read_text())
    assert "instance_only_alpha0" in VARIANTS
    assert LEGACY_VARIANTS["instance_only_alpha0"] == "instance_only_alpha0"
    assert config["protocol"]["alpha"] == 0.0
    assert HierarchicalDistillLoss(weak_weight=0.0).weak_weight == 0.0


def test_label_simplex_cli_and_config_are_eeg_only() -> None:
    config = yaml.safe_load((ROOT / "configs/stage2_label_simplex.yaml").read_text())
    stage3 = yaml.safe_load((ROOT / "configs/stage3_label_simplex.yaml").read_text())
    assert "label_simplex" in VARIANTS
    assert LEGACY_VARIANTS["label_simplex"] == "label_simplex"
    assert config["protocol"]["emg_access"] == "prohibited"
    assert config["protocol"]["label_loss_weight"] == 1.0
    assert stage3["protocol"]["variant"] == "label_simplex"
    assert stage3["protocol"]["emg_access"] == "prohibited"
    assert stage3["protocol"]["final_test_policy"] == "once_after_selected_checkpoint_reload"


def test_frozen_stage3_config_e() -> None:
    config = yaml.safe_load((ROOT / "configs/stage3.yaml").read_text())["protocol"]
    assert config["encoder_learning_rate"] == 3e-6
    assert config["classifier_learning_rate"] == 1e-4
    assert config["dropout"] == 0.5
    assert config["label_smoothing"] == 0.1
    assert config["epochs"] == 100
    assert config["final_test_policy"] == "once_after_selected_checkpoint_reload"


def test_eegnet_82_architecture() -> None:
    values = architecture_dict(EEGNet82())
    assert (values["F1"], values["D"], values["F2"]) == (8, 2, 16)
    assert values["input_shape"] == [60, 256]


def test_hubert_documentation_is_consistent() -> None:
    text = "\n".join(
        (ROOT / path).read_text()
        for path in ("README.md", "docs/DATA.md", "docs/REPRODUCIBILITY.md")
    )
    assert "HuBERT Base" in text
    assert "ninth Transformer block" in text
    assert "AttentionPooling is not" in text


def test_readme_script_and_config_entries_exist() -> None:
    text = (ROOT / "README.md").read_text()
    referenced = set(re.findall(r"(?:scripts|configs)/[A-Za-z0-9_.-]+", text))
    assert "configs/stage2_label_simplex.yaml" in referenced
    assert "scripts/generate_protocol_manifests.py" in referenced
    missing = sorted(path for path in referenced if not (ROOT / path).is_file())
    assert missing == []
