"""Traceable subject-level aggregation of recorded Stage 3 results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ALIASES = {"shuffled_emg_matched": "unrestricted_shuffle", "within_subject_class_deranged": "within_class_shuffle"}
REQUIRED_VARIANTS = ("full", "no_distill", "unrestricted_shuffle", "within_class_shuffle")
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

def write_aggregate(frame, output):
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
    (output / "provenance.json").write_text(json.dumps({"input_rows":len(frame),"subject_level_unit":30,"seeds":sorted(SEEDS),"folds":5,"variants":REQUIRED_VARIANTS}, indent=2) + "\n")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--results-root"); source.add_argument("--frozen-stage3-csv")
    parser.add_argument("--within-class-results-root"); parser.add_argument("--frozen-stage2-csv"); parser.add_argument("--output",required=True); args = parser.parse_args(argv)
    if args.results_root:
        if args.within_class_results_root: raise SystemExit("--within-class-results-root is only valid with --frozen-stage3-csv")
        frame = recorded_results(args.results_root)
    else:
        if not args.within_class_results_root: raise SystemExit("--within-class-results-root is required with --frozen-stage3-csv")
        within = recorded_results(args.within_class_results_root); within = within[within.variant == "within_class_shuffle"]
        frame = pd.concat([frozen_manifest(args.frozen_stage3_csv), within], ignore_index=True)
    require_complete(frame); write_aggregate(frame,args.output)
    if args.frozen_stage2_csv:
        aggregate_stage2(args.frozen_stage2_csv, args.within_class_results_root or args.results_root, args.output)
    print(args.output)
if __name__ == "__main__": main()
