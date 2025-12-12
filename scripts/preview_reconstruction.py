#!/usr/bin/env python

"""
Quick visualization utility for RBYRCT experiments.

Shows:
  - Ground-truth phantom (2D)
  - MART reconstruction (reshaped if needed)
  - Optional difference image

Usage:
  python scripts/preview_reconstruction.py \
    --run-id topas_run_0001 \
    --exp-id exp_mart_baseline_full_001
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_and_align(phantom_path: Path, recon_path: Path):
    phantom = np.load(phantom_path)
    recon = np.load(recon_path)

    if phantom.ndim != 2:
        raise ValueError(f"Expected 2D phantom, got shape {phantom.shape}")

    # If recon is flattened, reshape it to match phantom
    if recon.ndim == 1:
        if recon.size != phantom.size:
            raise ValueError(
                f"Recon size {recon.size} does not match phantom size {phantom.size}"
            )
        recon = recon.reshape(phantom.shape)

    if recon.shape != phantom.shape:
        raise ValueError(
            f"Recon shape {recon.shape} does not match phantom shape {phantom.shape}"
        )

    return phantom, recon


def plot_side_by_side(phantom, recon, show_diff=True, title=None):
    ncols = 3 if show_diff else 2
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    if ncols == 1:
        axes = [axes]

    im0 = axes[0].imshow(phantom, cmap="gray")
    axes[0].set_title("Ground Truth Phantom")
    axes[0].axis("off")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(recon, cmap="gray")
    axes[1].set_title("MART Reconstruction")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    if show_diff:
        diff = recon - phantom
        vmax = max(abs(diff.min()), abs(diff.max()))
        im2 = axes[2].imshow(diff, cmap="bwr", vmin=-vmax, vmax=vmax)
        axes[2].set_title("Difference (Recon − GT)")
        axes[2].axis("off")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Preview phantom vs MART reconstruction side by side."
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="topas_run_* ID (for phantom.npy)",
    )
    parser.add_argument(
        "--exp-id",
        required=True,
        help="Experiment ID under experiments/ (for volume_recon.npy)",
    )
    parser.add_argument(
        "--no-diff",
        action="store_true",
        help="Disable difference image.",
    )
    args = parser.parse_args()

    phantom_path = Path("data/raw/topas_runs") / args.run_id / "phantom.npy"
    recon_path = Path("experiments") / args.exp_id / "volume_recon.npy"

    if not phantom_path.is_file():
        raise FileNotFoundError(f"Phantom not found: {phantom_path}")
    if not recon_path.is_file():
        raise FileNotFoundError(f"Reconstruction not found: {recon_path}")

    phantom, recon = load_and_align(phantom_path, recon_path)

    title = f"{args.exp_id}  |  {args.run_id}"
    plot_side_by_side(
        phantom,
        recon,
        show_diff=not args.no_diff,
        title=title,
    )


if __name__ == "__main__":
    main()

