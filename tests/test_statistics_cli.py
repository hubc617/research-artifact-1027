import itertools
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/paired_subject_statistics.py"


@pytest.fixture
def frame():
    return pd.DataFrame(
        (
            (f"S{subject:02d}", paradigm, variant, seed, fold,
             40 + fold + (subject / 10 if variant == "full" else 0))
            for subject, paradigm, variant, seed, fold in itertools.product(
                range(1, 31), ("overt", "silent_1001"), ("full", "no_distill"),
                (42, 2026, 3407), range(5),
            )
        ),
        columns=["subject", "paradigm", "variant", "seed", "fold", "balanced_accuracy"],
    )


def run_statistics(tmp_path, frame):
    source, target = tmp_path / "input.csv", tmp_path / "output.csv"
    frame.to_csv(source, index=False)
    process = subprocess.run(
        [sys.executable, str(SCRIPT), "--input", str(source), "--output", str(target),
         "--left", "full", "--right", "no_distill",
         "--bootstrap-draws", "100", "--sign-flip-draws", "200"],
        capture_output=True, text=True,
    )
    return process, target


@pytest.mark.parametrize("task_level", [True, False])
def test_paired_statistics_complete_grid(tmp_path, frame, task_level):
    if not task_level:
        frame = frame.groupby(["subject", "paradigm", "variant"], as_index=False).balanced_accuracy.mean()
    process, target = run_statistics(tmp_path, frame)
    assert process.returncode == 0, process.stderr
    rows = pd.read_csv(target)
    assert list(rows.paradigm) == ["overt", "silent_1001"]
    assert list(rows.n_subjects) == [30, 30]
    assert list(rows.positive_subjects) == [30, 30]
    assert list(rows.mean_difference_pp) == pytest.approx([1.55, 1.55])


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        (lambda f: f.drop(index=f.index[f.variant == "no_distill"]), "overt/no_distill"),
        (lambda f: f.assign(balanced_accuracy=f.balanced_accuracy.mask(f.index == 0)), "must be finite"),
        (lambda f: f.assign(balanced_accuracy=f.balanced_accuracy.mask(f.index == 0, float("inf"))), "must be finite"),
        (lambda f: f.assign(fold=f.fold.mask(f.index == 0, 1)), "duplicates=1"),
        (lambda f: f.assign(fold=f.fold.astype(float).mask(f.index == 0, 0.5)), "expected integer"),
        (lambda f: f.drop(columns="seed"), "supply both seed and fold"),
        (lambda f: f.drop(index=f.index[f.paradigm == "silent_1001"]), "silent_1001/full"),
    ],
)
def test_paired_statistics_reject_bad_input(tmp_path, frame, invalid, message):
    process, target = run_statistics(tmp_path, invalid(frame))
    assert process.returncode != 0
    assert message in process.stderr
    assert not target.exists()
