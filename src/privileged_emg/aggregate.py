"""Traceable subject-level aggregation of recorded Stage 3 results."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

ALIASES = {"shuffled_emg_matched": "unrestricted_shuffle", "within_subject_class_deranged": "within_class_shuffle"}
REQUIRED_VARIANTS = ("full", "no_distill", "unrestricted_shuffle", "within_class_shuffle")
SIX_VARIANTS = (*REQUIRED_VARIANTS, "instance_only_alpha0", "label_simplex")
SUBJECTS = {f"S{i:02d}" for i in range(1, 31)}
PARADIGMS, SEEDS = {"overt", "silent_1001"}, {42, 2026, 3407}

def canonical(value): return ALIASES.get(value, value)
def bootstrap(x, rng, n=50000):
    x = np.asarray(x, float); draws = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return float(x.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))
def signflip(x, rng, n=200000):
    x, observed, exceed = np.asarray(x, float), abs(np.mean(x)), 0
    for start in range(0, n, 20000):
        signs = rng.choice((-1., 1.), size=(min(20000, n-start), len(x)))
        exceed += int((np.abs((signs*x).mean(1)) >= observed).sum())
    return (exceed + 1) / (n + 1)

def recorded_results(root):
    rows = []
    for path in Path(root).glob("fold*/seed*/stage2_*/stage3/S*/*/result.json"):
        result = json.loads(path.read_text())
        if "test_balanced_accuracy" in result:
            rows.append({"subject": result["subject"], "paradigm": result["paradigm"], "fold": int(result["fold"]), "seed": int(result["seed"]), "variant": canonical(result.get("variant", path.parents[3].name.removeprefix("stage2_"))), "balanced_accuracy": float(result["test_balanced_accuracy"]), "source_result": str(path)})
    return pd.DataFrame(rows)

def frozen_manifest(path):
    frame = pd.read_csv(path); required = {"subject", "paradigm_raw", "fold", "seed", "variant", "test_balanced_accuracy"}
    if missing := required - set(frame.columns): raise ValueError(f"{path}: missing {sorted(missing)}")
    return pd.DataFrame({"subject": frame.subject, "paradigm": frame.paradigm_raw, "fold": frame.fold.astype(int), "seed": frame.seed.astype(int), "variant": frame.variant.map(canonical), "balanced_accuracy": frame.test_balanced_accuracy.astype(float), "source_result": str(path)})

def aggregate_stage2(frozen_csv, within_root, output):
    frozen = pd.read_csv(frozen_csv)
    required = {"fold", "seed", "variant", "best_val_balanced_accuracy"}
    if missing := required - set(frozen.columns): raise ValueError(f"{frozen_csv}: missing {sorted(missing)}")
    rows = [{"fold":int(r.fold), "seed":int(r.seed), "variant":canonical(r.variant), "balanced_accuracy":float(r.best_val_balanced_accuracy)} for r in frozen.itertuples()]
    for path in Path(within_root).glob("fold*/seed*/stage2_within_subject_class_deranged/result.json"):
        result = json.loads(path.read_text())
        rows.append({"fold":int(result["fold"]), "seed":int(result["seed"]), "variant":"within_class_shuffle", "balanced_accuracy":float(result["best_val_balanced_accuracy"])})
    frame = pd.DataFrame(rows)
    expected = {(fold, seed, variant) for fold in range(5) for seed in SEEDS for variant in REQUIRED_VARIANTS}
    observed = set(zip(frame.fold, frame.seed, frame.variant))
    if observed != expected or len(frame) != len(expected): raise ValueError(f"incomplete Stage 2 grid: rows={len(frame)}, expected={len(expected)}, missing={len(expected-observed)}")
    frame.groupby("variant",as_index=False).balanced_accuracy.mean().to_csv(Path(output) / "stage2_validation_summary.csv",index=False)

def require_complete(frame):
    expected = {(s,p,f,seed,v) for s in SUBJECTS for p in PARADIGMS for f in range(5) for seed in SEEDS for v in REQUIRED_VARIANTS}
    observed = set(zip(frame.subject, frame.paradigm, frame.fold, frame.seed, frame.variant))
    if missing := expected-observed: raise ValueError(f"incomplete result grid: rows={len(frame)}, expected={len(expected)}, missing={len(missing)}")
    if len(frame) != len(expected): raise ValueError(f"duplicate or unexpected rows: rows={len(frame)}, expected={len(expected)}")

def write_aggregate(frame, output, variants=REQUIRED_VARIANTS):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    subject = frame.groupby(["subject", "paradigm", "variant"], as_index=False).balanced_accuracy.mean()
    subject.groupby(["paradigm", "variant"], as_index=False).balanced_accuracy.mean().to_csv(output / "table1_reconstructed.csv", index=False)
    rows, rng, perm = [], np.random.default_rng(20260814), np.random.default_rng(20260815)
    comparisons = (("full","no_distill"),("unrestricted_shuffle","no_distill"),("within_class_shuffle","no_distill"),("within_class_shuffle","unrestricted_shuffle"),("full","within_class_shuffle"))
    for paradigm in sorted(PARADIGMS):
        pivot = subject[subject.paradigm == paradigm].pivot(index="subject", columns="variant", values="balanced_accuracy")
        for left,right in comparisons:
            x = (pivot[left]-pivot[right]).dropna().to_numpy()
            if len(x) != 30: raise AssertionError(f"{paradigm} {left}-{right}: n_subjects={len(x)}")
            mean,low,high = bootstrap(x,rng); rows.append({"paradigm":paradigm,"comparison":f"{left}_minus_{right}","n_subjects":len(x),"mean_pp":mean,"ci95_low":low,"ci95_high":high,"positive_subjects":int((x>0).sum()),"signflip_p":signflip(x,perm)})
    pd.DataFrame(rows).to_csv(output / "table2_paired_subject_statistics.csv", index=False)
    (output / "provenance.json").write_text(json.dumps({"input_rows":len(frame),"subject_level_unit":30,"seeds":sorted(SEEDS),"folds":5,"variants":variants}, indent=2) + "\n")

def _read_object(raw, source):
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"{source}: invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{source}: expected a JSON object")
    return value


def _task_key(task, stage, source):
    key = ["variant", "seed", "fold"]
    if stage == 3:
        key += ["subject", "paradigm"]
    if type(task.get("stage")) is not int or task["stage"] != stage:
        raise ValueError(f"{source}: expected Stage {stage} task")
    if any(type(task.get(field)) is not int for field in ("seed", "fold")):
        raise ValueError(f"{source}: seed and fold must be integers")
    if any(not isinstance(task.get(field), str) for field in key if field not in ("seed", "fold")):
        raise ValueError(f"{source}: expected string identity fields {key}")
    return tuple(task[field] for field in key)


def _result_path(task, source):
    directory = task.get("out_dir")
    if not isinstance(directory, str) or not directory.strip():
        raise ValueError(f"{source}: missing or invalid out_dir")
    return (Path(directory) / "result.json").resolve()


def manifest_results(manifests, stage, variants):
    cells = {(v, s, f) for v in variants for s in SEEDS for f in range(5)}
    expected = cells if stage == 2 else {
        (*cell, subject, paradigm)
        for cell in cells for subject in SUBJECTS for paradigm in PARADIGMS
    }
    tasks, paths = {}, set()
    for manifest in manifests:
        with manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                source = f"{manifest}:{line_number}"
                task = _read_object(line, source)
                key = _task_key(task, stage, source)
                if key not in expected:
                    raise ValueError(f"{source}: unexpected Stage {stage} key {key}")
                if key in tasks:
                    raise ValueError(f"{source}: duplicate Stage {stage} key {key}")
                path = _result_path(task, source)
                if path in paths:
                    raise ValueError(f"{source}: duplicate result path {path}")
                tasks[key] = path
                paths.add(path)
    if missing := expected - tasks.keys():
        raise ValueError(
            f"incomplete Stage {stage} manifest grid: {len(tasks)}/{len(expected)} tasks, "
            f"missing {len(missing)} including {sorted(missing)[:3]}"
        )

    metric = "best_val_balanced_accuracy" if stage == 2 else "test_balanced_accuracy"
    rows = []
    for key, path in tasks.items():
        if not path.is_file():
            raise ValueError(f"{path}: missing result for Stage {stage} task {key}")
        result = _read_object(path.read_text(encoding="utf-8"), path)
        observed = _task_key(result, stage, path)
        if observed != key or _result_path(result, path) != path:
            raise ValueError(f"{path}: result identity mismatch, expected task {key}, got {observed}")
        accuracy = result.get(metric)
        if type(accuracy) not in (int, float) or not 0 <= accuracy <= 100 or not math.isfinite(accuracy):
            raise ValueError(f"{path}: {metric} must be finite and in 0–100 percent")
        variant, seed, fold, *subject_paradigm = key
        row = {
            "variant": variant, "seed": seed, "fold": fold,
            "balanced_accuracy": float(accuracy), "source_result": str(path),
        }
        if stage == 3:
            row["subject"], row["paradigm"] = subject_paradigm
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results-root")
    source.add_argument("--frozen-stage3-csv")
    source.add_argument(
        "--stage3-manifest", action="append", type=Path,
        help="Stage 3 JSONL from generate_protocol_manifests.py. Repeat to combine batches.",
    )
    parser.add_argument(
        "--stage2-manifest", action="append", type=Path,
        help="Optional matching Stage 2 JSONL. Repeat to combine batches.",
    )
    parser.add_argument(
        "--grid", choices=("four", "six"),
        help="Required for manifest input. Stage 3: 3600/5400 tasks. Stage 2: 60/90 tasks.",
    )
    parser.add_argument("--within-class-results-root")
    parser.add_argument("--frozen-stage2-csv")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.stage3_manifest:
        if not args.grid or args.within_class_results_root or args.frozen_stage2_csv:
            parser.error("manifest input requires --grid and cannot be mixed with legacy input")
        variants = REQUIRED_VARIANTS if args.grid == "four" else SIX_VARIANTS
        stage3 = manifest_results(args.stage3_manifest, 3, variants)
        stage2 = manifest_results(args.stage2_manifest, 2, variants) if args.stage2_manifest else None
        write_aggregate(stage3, args.output, variants)
        if stage2 is not None:
            stage2.groupby("variant", as_index=False).balanced_accuracy.mean().to_csv(
                args.output / "stage2_validation_summary.csv", index=False,
            )
    else:
        if args.grid or args.stage2_manifest:
            parser.error("--grid and --stage2-manifest require --stage3-manifest")
        if args.results_root:
            if args.within_class_results_root:
                parser.error("--within-class-results-root is only valid with --frozen-stage3-csv")
            frame = recorded_results(args.results_root)
        else:
            if not args.within_class_results_root:
                parser.error("--within-class-results-root is required with --frozen-stage3-csv")
            within = recorded_results(args.within_class_results_root)
            within = within[within.variant == "within_class_shuffle"]
            frame = pd.concat([frozen_manifest(args.frozen_stage3_csv), within], ignore_index=True)
        require_complete(frame)
        write_aggregate(frame, args.output)
        if args.frozen_stage2_csv:
            aggregate_stage2(
                args.frozen_stage2_csv, args.within_class_results_root or args.results_root, args.output,
            )
    print(args.output)


if __name__ == "__main__":
    main()
