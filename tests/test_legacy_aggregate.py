import csv
import json
from itertools import product
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_paper_results.py"
SEED_FOLDS = tuple(product((42, 2026, 3407), range(5)))
VARIANTS = ("full", "no_distill", "shuffled_emg_matched", "within_subject_class_deranged")
STAGE3_METRIC = "test_balanced_accuracy"
STAGE2_METRIC = "best_val_balanced_accuracy"


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def result_path(root, row, *, stage3):
    task = root / f"fold{row['fold']}" / f"seed{row['seed']}" / f"stage2_{row['variant']}"
    if stage3:
        task = task / "stage3" / row["subject"] / row["paradigm_raw"]
    return task / "result.json"


def write_result(path, row):
    data = dict(row)
    if "paradigm_raw" in data:
        data["paradigm"] = data.pop("paradigm_raw")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def fixture_inputs(root):
    frozen_stage3, frozen_stage2, within = (
        root / "frozen-stage3.csv", root / "frozen-stage2.csv", root / "within",
    )
    stage3_rows, stage2_rows = [], []
    for variant_index, variant in enumerate(VARIANTS):
        for seed, fold in SEED_FOLDS:
            for subject_number, paradigm in product(range(1, 31), ("overt", "silent_1001")):
                row = {
                    "subject": f"S{subject_number:02d}", "paradigm_raw": paradigm,
                    "variant": variant, "seed": seed, "fold": fold,
                    STAGE3_METRIC: (100, 0, 30, 40)[variant_index],
                }
                if variant_index == 3:
                    write_result(result_path(within, row, stage3=True), row)
                else:
                    stage3_rows.append(row)
            row = dict(variant=variant, seed=seed, fold=fold)
            row[STAGE2_METRIC] = (50, 30, 40, 60)[variant_index]
            if variant_index == 3:
                write_result(result_path(within, row, stage3=False), row)
            else:
                stage2_rows.append(row)
    write_csv(frozen_stage3, stage3_rows)
    write_csv(frozen_stage2, stage2_rows)
    return frozen_stage3, frozen_stage2, within, stage3_rows, stage2_rows


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    return fixture_inputs(tmp_path_factory.mktemp("legacy-synthetic"))


def run_aggregate(output, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args), "--output", str(output)],
        capture_output=True, text=True,
    )


def assert_rejected(output, process, *details):
    assert process.returncode != 0, process.stdout
    for detail in details:
        assert detail in process.stderr
    assert not output.exists()


def test_legacy_frozen_results_accept_bounds_and_keep_statistics(inputs, tmp_path):
    frozen_stage3, frozen_stage2, within, _, _ = inputs
    output = tmp_path / "summary"
    process = run_aggregate(
        output, "--frozen-stage3-csv", frozen_stage3, "--within-class-results-root", within,
        "--frozen-stage2-csv", frozen_stage2,
    )
    assert process.returncode == 0, process.stderr
    means = pd.read_csv(output / "table1_reconstructed.csv")
    assert sorted(means.balanced_accuracy) == [0, 0, 30, 30, 40, 40, 100, 100]
    stage2_means = pd.read_csv(output / "stage2_validation_summary.csv")
    assert sorted(stage2_means.balanced_accuracy) == [30, 40, 50, 60]
    paired = pd.read_csv(output / "table2_paired_subject_statistics.csv")
    main_pair = paired[paired.comparison == "full_minus_no_distill"]
    assert list(main_pair.n_subjects) == list(main_pair.positive_subjects) == [30, 30]
    for column in ("mean_pp", "ci95_low", "ci95_high"):
        assert list(main_pair[column]) == [100, 100]
    assert list(main_pair.signflip_p) == pytest.approx([1 / 200_001] * 2)


@pytest.mark.parametrize("value", ["", "NaN", "Inf", "-Inf", "unmeasured", "-0.01", "100.01"])
def test_legacy_frozen_stage3_rejects_one_invalid_cell(inputs, tmp_path, value):
    _, _, within, rows, _ = inputs
    frozen = tmp_path / "frozen-invalid.csv"
    write_csv(frozen, [{**rows[0], STAGE3_METRIC: value}, *rows[1:]])
    output = tmp_path / "summary"
    process = run_aggregate(
        output, "--frozen-stage3-csv", frozen, "--within-class-results-root", within,
    )
    assert_rejected(output, process, STAGE3_METRIC, "S01", "overt", "full", "42", "0")


@pytest.mark.parametrize("problem", ["missing", "duplicate", "fractional fold"])
def test_legacy_frozen_stage3_checks_full_unique_grid(inputs, tmp_path, problem):
    _, _, within, rows, _ = inputs
    frozen = tmp_path / "frozen-invalid.csv"
    if problem == "missing":
        changed, detail = rows[1:], "missing"
    elif problem == "duplicate":
        changed, detail = [rows[1], *rows[1:]], "duplicate"
    else:
        changed, detail = [{**rows[0], "fold": 0.5}, *rows[1:]], "fold"
    write_csv(frozen, changed)
    output = tmp_path / "summary"
    process = run_aggregate(
        output, "--frozen-stage3-csv", frozen, "--within-class-results-root", within,
    )
    assert_rejected(output, process, detail, "S01")


def test_legacy_results_root_rejects_missing_metric(tmp_path):
    root = tmp_path / "recorded"
    first = {
        "subject": "S01", "paradigm_raw": "overt", "variant": "full", "seed": 42, "fold": 0,
    }
    write_result(result_path(root, first, stage3=True), first)
    output = tmp_path / "summary"
    assert_rejected(
        output, run_aggregate(output, "--results-root", root),
        STAGE3_METRIC, "S01", "overt", "full", "42", "0",
    )


def test_legacy_frozen_stage2_error_leaves_no_stage3_output(inputs, tmp_path):
    frozen_stage3, _, within, _, rows = inputs
    frozen_stage2 = tmp_path / "frozen-invalid.csv"
    write_csv(frozen_stage2, [{**rows[0], STAGE2_METRIC: ""}, *rows[1:]])
    output = tmp_path / "summary"
    process = run_aggregate(
        output, "--frozen-stage3-csv", frozen_stage3, "--within-class-results-root", within,
        "--frozen-stage2-csv", frozen_stage2,
    )
    assert_rejected(output, process, STAGE2_METRIC, "full", "42", "0")
