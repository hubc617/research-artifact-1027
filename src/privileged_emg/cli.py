"""Public command-line interface for the validated three-stage pipeline."""
from __future__ import annotations
import argparse, json, platform, subprocess, sys
from pathlib import Path
import yaml
from . import chain
from . import common
from .splits import generate_splits

VARIANTS = (
    "full",
    "no_distill",
    "unrestricted_shuffle",
    "within_class_shuffle",
    "instance_only_alpha0",
    "label_simplex",
)
LEGACY_VARIANTS = {
    "full": "full",
    "no_distill": "no_distill",
    "unrestricted_shuffle": "shuffled_emg_matched",
    "within_class_shuffle": "within_subject_class_deranged",
    "instance_only_alpha0": "instance_only_alpha0",
    "label_simplex": "label_simplex",
}

def load_config(path):
    with Path(path).open() as f: return yaml.safe_load(f) or {}

def runtime_meta():
    try: commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception: commit="uncommitted"
    return {"python":sys.version,"platform":platform.platform(),"torch":__import__("torch").__version__,"git_commit":commit}

def run_train(args):
    cfg=load_config(args.config); paths=cfg.get("paths",{}); data_root=args.data_root or paths.get("data_root"); split_root=args.split_root or paths.get("split_root")
    if not data_root or not split_root: raise SystemExit("data_root and split_root are required via CLI or config")
    out=Path(args.output).resolve()
    if out.exists() and any(out.iterdir()) and not args.resume: raise SystemExit(f"refusing to reuse non-empty output without --resume: {out}")
    if out.exists() and any(out.iterdir()) and args.resume:
        result=out/"result.json"
        if result.exists(): print(f"resume: completed result already exists: {result}"); return
    common.configure(data_root); chain.configure(split_root)
    task={"seed":args.seed,"fold":args.fold,"epochs":args.epochs,"out_dir":str(out),"resolved_config":cfg,"runtime":runtime_meta()}
    if args.stage == "stage1": chain.stage1(task)
    elif args.stage == "stage2":
        if args.variant not in ("label_simplex", "no_distill") and not args.stage1_checkpoint:
            raise SystemExit("--stage1-checkpoint is required for EMG-teacher Stage 2 variants")
        if args.variant in ("label_simplex", "no_distill") and args.stage1_checkpoint:
            raise SystemExit(f"{args.variant} prohibits --stage1-checkpoint")
        alpha = 0.0 if args.variant == "instance_only_alpha0" else 0.3
        task.update({
            "stage1_ckpt": None if args.variant in ("label_simplex", "no_distill") else args.stage1_checkpoint,
            "variant": LEGACY_VARIANTS[args.variant],
            "weak_weight": None if args.variant == "label_simplex" else alpha,
            "alpha": None if args.variant == "label_simplex" else alpha,
            "target": (
                "fixed_regular_simplex_by_lexical_label"
                if args.variant == "label_simplex"
                else (
                    "normalized_paired_emg_representation_only"
                    if args.variant == "instance_only_alpha0"
                    else "normalized_paired_emg_plus_class_prototype"
                )
            ),
        })
        chain.stage2(task)
    else:
        if not args.stage2_checkpoint or not args.subject or not args.paradigm: raise SystemExit("Stage 3 requires --stage2-checkpoint, --subject, and --paradigm")
        task.update({"stage2_ckpt":args.stage2_checkpoint,"subject":args.subject,"paradigm":args.paradigm,"variant":args.variant})
        chain.stage3(task)

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); subs=ap.add_subparsers(dest="command",required=True)
    split=subs.add_parser("prepare-splits",help="generate deterministic five-fold splits from local data"); split.add_argument("--data-root",required=True); split.add_argument("--output",required=True)
    train=subs.add_parser("train",help="run a real Stage 1, 2, or 3 training task")
    train.add_argument("--stage",choices=("stage1","stage2","stage3"),required=True); train.add_argument("--config",required=True); train.add_argument("--output",required=True); train.add_argument("--data-root"); train.add_argument("--split-root"); train.add_argument("--seed",type=int,default=42); train.add_argument("--fold",type=int,default=0); train.add_argument("--epochs",type=int,required=True); train.add_argument("--variant",choices=VARIANTS,default="full"); train.add_argument("--stage1-checkpoint"); train.add_argument("--stage2-checkpoint"); train.add_argument("--subject"); train.add_argument("--paradigm",choices=("overt","silent_1001")); train.add_argument("--resume",action="store_true")
    args=ap.parse_args(argv)
    if args.command=="prepare-splits": common.configure(args.data_root); print(generate_splits(args.output))
    else: run_train(args)
if __name__=="__main__": main()
