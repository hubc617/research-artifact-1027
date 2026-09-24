#!/usr/bin/env python3
"""Render the grayscale-safe two-panel Fig. 2 from audited channel summaries."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from privileged_emg.secondary import EEG_CHANNELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-summary", type=Path, required=True)
    parser.add_argument("--annotation-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grayscale-output", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.channel_summary)
    required = {"paradigm", "channel", "full_minus_no_distill"}
    if missing := required - set(frame):
        raise ValueError(f"missing columns: {sorted(missing)}")
    if args.output.exists():
        raise FileExistsError(args.output)
    info = mne.create_info(EEG_CHANNELS, 256, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"), match_case=False)
    fig, axes = plt.subplots(1, 2, figsize=(5.65, 2.7))
    annotations = None
    if args.annotation_summary:
        annotations = pd.read_csv(args.annotation_summary).set_index("paradigm")
        needed = {"shift_pp", "positive_subjects", "total_subjects"}
        if missing := needed - set(annotations):
            raise ValueError(f"annotation summary missing columns: {sorted(missing)}")
    image = None
    for ax, paradigm, title in zip(axes, ("overt", "silent_1001"), ("Overt", "Silent")):
        part = frame[frame.paradigm == paradigm].set_index("channel")
        if set(part.index) != set(EEG_CHANNELS) or len(part) != 60:
            raise ValueError(f"{paradigm}: expected exactly 60 unique channels")
        values = np.asarray([part.loc[channel, "full_minus_no_distill"] for channel in EEG_CHANNELS])
        image, contours = mne.viz.plot_topomap(
            values, info, axes=ax, show=False, cmap="RdBu_r",
            vlim=(-0.005, 0.005), contours=(-0.0035, -0.002, 0.002, 0.0035),
        )
        if contours is not None:
            contours.set_linestyles(["dashed", "dashed", "solid", "solid"])
        ax.set_title(title)
        if annotations is not None:
            row = annotations.loc[paradigm]
            ax.text(
                0.5, -0.08,
                f"peripheral shift {float(row.shift_pp):+.2f} pp; "
                f"{int(row.positive_subjects)}/{int(row.total_subjects)} subjects",
                transform=ax.transAxes, ha="center", va="top", fontsize=8,
            )
    fig.colorbar(image, ax=axes, fraction=0.04, pad=0.04, label="Full - NoDistill")
    fig.savefig(args.output)
    plt.close(fig)
    if b"/Subtype /Type3" in args.output.read_bytes():
        raise RuntimeError("Type-3 font detected")
    if args.grayscale_output:
        if args.grayscale_output.exists():
            raise FileExistsError(args.grayscale_output)
        from PIL import Image
        png = args.output.with_suffix(".png")
        # Re-rendering a PDF through external tools is intentionally avoided;
        # request a PNG output when a raster grayscale preview is required.
        if args.output.suffix.lower() != ".png":
            raise ValueError("grayscale preview requires --output ending in .png")
        Image.open(png).convert("L").save(args.grayscale_output)


if __name__ == "__main__":
    main()
