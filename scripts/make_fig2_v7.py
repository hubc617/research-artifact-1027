#!/usr/bin/env python3
"""Render the compact v7 two-panel Fig. 2 from channel and annotation summaries."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
    "axes.unicode_minus": True,
})
import matplotlib.pyplot as plt
import mne
from mne.channels.layout import _find_topomap_coords
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from privileged_emg.secondary import EEG_CHANNELS

PANELS = {"overt": "Overt", "silent_1001": "Silent"}
COLOR_LIMIT = 0.005
CONTOUR_LEVELS = [-0.0035, -0.0020, 0.0020, 0.0035]
ELECTRODE_LABELS = {
    "FZ": ("Fz", (0, 6)), "CZ": ("Cz", (0, 8)),
    "CPZ": ("CPz", (0, -1)), "PZ": ("Pz", (0, -7)),
    "T7": ("T7", (-10, 0)), "T8": ("T8", (10, 0)),
}


def read_summary(path, columns):
    frame = pd.read_csv(path)
    if missing := set(columns) - set(frame):
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if set(frame.paradigm) != set(PANELS):
        raise ValueError(f"{path}: expected overt and silent_1001")
    return frame


def read_channel_vectors(path):
    frame = read_summary(path, ("paradigm", "channel", "full_minus_no_distill"))
    vectors = {}
    for paradigm in PANELS:
        part = frame[frame.paradigm == paradigm].set_index("channel")
        if not part.index.is_unique or set(part.index) != set(EEG_CHANNELS):
            raise ValueError(f"{path}: {paradigm} needs exactly 60 unique channels")
        values = part.loc[EEG_CHANNELS, "full_minus_no_distill"].to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.abs(values).max() > 1.0:
            raise ValueError(f"{path}: {paradigm} differences must be finite fractions in [-1, 1]")
        # The v7 renderer received float32 channel differences.
        vectors[paradigm] = values.astype(np.float32).astype(float)
    return vectors


def read_annotations(path):
    names = ("shift_pp", "positive_subjects", "total_subjects")
    frame = read_summary(path, ("paradigm", *names)).set_index("paradigm")
    if not frame.index.is_unique:
        raise ValueError(f"{path}: expected one annotation per paradigm")
    labels = {}
    for paradigm in PANELS:
        shift, positive, total = frame.loc[paradigm, list(names)].to_numpy(dtype=float)
        if (
            not np.isfinite(shift) or abs(shift) > 100
            or not positive.is_integer() or not total.is_integer()
            or not 0 <= positive <= total or total <= 0
        ):
            raise ValueError(f"{path}: invalid shift or subject counts for {paradigm}")
        labels[paradigm] = f"{shift:+.2f} pp  ·  {int(positive)}/{int(total)}"
    return labels


def render(vectors, annotations, output):
    info = mne.create_info(EEG_CHANNELS, 256, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"), match_case=False)
    xy = _find_topomap_coords(info, np.arange(len(EEG_CHANNELS)), sphere=None)

    fig = plt.figure(figsize=(5.65, 2.70), facecolor="white")
    grid = fig.add_gridspec(
        2, 3,
        width_ratios=[1.0, 1.0, 0.060], height_ratios=[1.0, 0.13],
        left=0.020, right=0.845, bottom=0.045, top=0.895, wspace=0.085, hspace=0.015,
    )
    try:
        for column, (paradigm, title) in enumerate(PANELS.items()):
            ax = fig.add_subplot(grid[0, column])
            image, contours = mne.viz.plot_topomap(
                vectors[paradigm], xy, axes=ax, show=False, cmap="RdBu_r",
                vlim=(-COLOR_LIMIT, COLOR_LIMIT), contours=CONTOUR_LEVELS,
                sensors=True, res=256, outlines="head", extrapolate="head",
                border="mean", image_interp="cubic",
            )
            if contours is not None:
                contours.set_linestyles([
                    "dashed" if level < 0 else "solid" for level in contours.levels
                ])
            for index, channel in enumerate(EEG_CHANNELS):
                if channel in ELECTRODE_LABELS:
                    name, offset = ELECTRODE_LABELS[channel]
                    ax.annotate(
                        name, xy[index], xytext=offset, textcoords="offset points",
                        ha="center", va="center", fontsize=15, fontweight="bold",
                        color="black", zorder=20,
                    )
            ax.set_title(title, fontsize=19, fontweight="bold", pad=3)
            if annotations:
                footer = fig.add_subplot(grid[1, column])
                footer.set_axis_off()
                footer.text(
                    0.5, 0.5, annotations[paradigm], transform=footer.transAxes,
                    ha="center", va="center", fontsize=15,
                )

        colorbar = fig.colorbar(image, cax=fig.add_subplot(grid[0, 2]))
        colorbar.set_ticks([-0.004, 0.0, 0.004])
        colorbar.ax.tick_params(labelsize=15, width=0.7, length=3, pad=2)
        colorbar.ax.set_title(r"$\Delta$", fontsize=15, pad=3)
        dpi = 400 if output.suffix.lower() == ".png" else None
        fig.savefig(output, facecolor="white", dpi=dpi)
    finally:
        plt.close(fig)
    if output.suffix.lower() == ".pdf" and b"/Subtype /Type3" in output.read_bytes():
        raise RuntimeError("Type-3 font detected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel-summary", type=Path, required=True)
    parser.add_argument("--annotation-summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grayscale-output", type=Path)
    args = parser.parse_args()
    for output in (args.output, args.grayscale_output):
        if output and output.exists():
            raise FileExistsError(output)
    if args.grayscale_output and (
        args.output.suffix.lower() != ".png" or args.grayscale_output.suffix.lower() != ".png"
        or args.output.resolve() == args.grayscale_output.resolve()
    ):
        parser.error("grayscale preview requires two distinct PNG outputs")
    vectors = read_channel_vectors(args.channel_summary)
    annotations = read_annotations(args.annotation_summary) if args.annotation_summary else {}
    render(vectors, annotations, args.output)
    if args.grayscale_output:
        from PIL import Image
        with Image.open(args.output) as image:
            image.convert("L").save(args.grayscale_output)


if __name__ == "__main__":
    main()
