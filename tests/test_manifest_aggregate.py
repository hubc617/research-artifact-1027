import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def run_cli(script, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *map(str, args)],
        capture_output=True, text=True,
    )


@pytest.fixture
def manifests(tmp_path):
    directory = tmp_path / "manifests"
    process = run_cli(
        "generate_protocol_manifests.py",
        "--data-root", tmp_path / "data", "--split-root", tmp_path / "splits",
        "--run-root", tmp_path / "runs", "--output-dir", directory,
    )
    assert process.returncode == 0, process.stderr
    return directory


def manifest_paths(directory, grid):
    suffix = "six_variant" if grid == "six" else "clean_main_four_variant"
    stage2_count, stage3_count = (90, 5400) if grid == "six" else (60, 3600)
    return directory / f"stage2_{suffix}_{stage2_count}.jsonl", directory / f"stage3_{suffix}_{stage3_count}.jsonl"


def save_fixture_results(manifest, metric):
    for line in manifest.read_text().splitlines():
        task = json.loads(line)
        path = Path(task["out_dir"]) / "result.json"
        path.parent.mkdir(parents=True)
        value = 40 + task["fold"] + (2 if task["variant"] == "full" else 0)
        path.write_text(json.dumps({**task, metric: value}))


@pytest.mark.parametrize("grid", ["four", "six"])
def test_aggregate_formal_manifests(manifests, tmp_path, grid):
    stage2, stage3 = manifest_paths(manifests, grid)
    save_fixture_results(stage2, "best_val_balanced_accuracy")
    save_fixture_results(stage3, "test_balanced_accuracy")
    output = tmp_path / "summary"
    args = ["--stage2-manifest", stage2, "--stage3-manifest", stage3]
    if grid == "six":
        args = []
        for stage, complete, core in zip((2, 3), (stage2, stage3), manifest_paths(manifests, "four")):
            standalone = manifests / f"stage{stage}_additional.jsonl"
            core_lines = set(core.read_text().splitlines())
            standalone.write_text("".join(
                line + "\n" for line in complete.read_text().splitlines() if line not in core_lines
            ))
            args.extend((f"--stage{stage}-manifest", core, f"--stage{stage}-manifest", standalone))
    process = run_cli("aggregate_paper_results.py", "--grid", grid, *args, "--output", output)
    assert process.returncode == 0, process.stderr
    means = pd.read_csv(output / "table1_reconstructed.csv")
    stage2_means = pd.read_csv(output / "stage2_validation_summary.csv")
    expected_variants = 6 if grid == "six" else 4
    assert len(means) == 2 * expected_variants and len(stage2_means) == expected_variants
    assert set(means.loc[means.variant == "full", "balanced_accuracy"]) == {44.0}
    assert set(means.loc[means.variant == "no_distill", "balanced_accuracy"]) == {42.0}
    assert len(pd.read_csv(output / "table2_paired_subject_statistics.csv")) == 10


@pytest.mark.parametrize(
    ("invalid", "message"),
    [
        ("duplicate", "duplicate Stage 3 key"),
        ("missing", "incomplete Stage 3 manifest grid"),
        ("out_dir", "duplicate result path"),
        ("noninteger", "seed and fold must be integers"),
        ("missing_result", "missing result for Stage 3 task"),
        ("mismatched_result", "result identity mismatch"),
        ("nonfinite_result", "must be finite"),
    ],
)
def test_aggregate_manifest_rejects_bad_input(manifests, tmp_path, invalid, message):
    _, stage3 = manifest_paths(manifests, "four")
    rows = [json.loads(line) for line in stage3.read_text().splitlines()]
    if invalid == "duplicate":
        rows[0] = rows[1]
    elif invalid == "missing":
        rows.pop(0)
    elif invalid == "out_dir":
        rows[0]["out_dir"] = rows[1]["out_dir"]
    elif invalid == "noninteger":
        rows[0]["fold"] = 0.5
    elif invalid in {"mismatched_result", "nonfinite_result"}:
        path = Path(rows[0]["out_dir"]) / "result.json"
        path.parent.mkdir(parents=True)
        result = {**rows[0], "test_balanced_accuracy": 42.0}
        if invalid == "mismatched_result":
            result["paradigm"] = rows[1]["paradigm"]
        else:
            result["test_balanced_accuracy"] = float("nan")
        path.write_text(json.dumps(result))
    stage3.write_text("\n".join(map(json.dumps, rows)) + "\n")
    output = tmp_path / "summary"
    process = run_cli(
        "aggregate_paper_results.py", "--grid", "four", "--stage3-manifest", stage3,
        "--output", output,
    )
    assert process.returncode != 0
    assert message in process.stderr
    assert not output.exists()
