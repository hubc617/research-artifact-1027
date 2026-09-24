#!/usr/bin/env python3
"""Run one deterministic manifest task with configurable data and split roots."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from privileged_emg import chain, common
from privileged_emg.common import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("stage1", "stage2", "stage3"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-root", required=True)
    args = parser.parse_args()
    common.configure(args.data_root)
    chain.configure(args.split_root)
    task = read_jsonl(args.manifest, args.index)
    task["_manifest_dir"] = str(Path(args.manifest).resolve().parent)
    {"stage1": chain.stage1, "stage2": chain.stage2, "stage3": chain.stage3}[args.kind](task)


if __name__ == "__main__":
    main()
