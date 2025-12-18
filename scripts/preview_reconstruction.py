#!/usr/bin/env python3

"""
Preview phantom vs reconstruction (and optional denoised) side by side.

Looks for:
  data/raw/topas_runs/<run-id>/phantom.npy
  experiments/<exp-id>/volume_recon.npy
  experiments/<exp-id>/volume_denoised.npy   (optional)

Usage:
  python scripts/preview_reconstruction.py --run-id topas_run_0001 --exp-id exp_mart_baseline_full_001
  python scripts/preview_reconstruction.py --run-id topas_run_0001 --exp-id exp_mart_baseline_full_001 --save
  python scripts/preview_reconstruction.py --run-id topas_run_0001 --exp-id exp_mart_baseline_full_001 --no-diff
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _load(path: Path) -> np.ndarray:
    return np.load(path)


def _align_to_phantom(phantom_2d: np.ndarray, arr: np.ndarray, name: str) -> np.ndarray:
    if phantom_2d.ndim != 2:
        raise ValueError(f"Expected 2D phantom, got shape {phantom_2d.shape}")

    if arr.ndim == 1:
        if arr.size != phantom_2d.size:
            raise ValueError(f"{name} size {arr.size} != phantom size {phantom_2d.size}")
        return arr.reshape(phantom_2d.shape)

    if arr.shape != phantom_2d.shape:
        raise ValueError(f"{name} shape {arr.shape} != phantom shape {phantom_2d.shape}")

    return arr


def _imshow(ax, img, title: str, cmap: str = "gray", vmin=None, vmax=None):
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    return im


def main():
    parser = argparse.ArgumentParser(description="Preview phantom vs recon/denoised images.")
    parser.add_argument("--run-id", required=True, help="topas_run_* ID (for phantom.npy)")
    parser.add_argument("--exp-id", required=True, help="Experiment ID under experiments/")
    parser.add_argument("--denoised-name", default="volume_denoised.npy",
                        help="Filename for denoised output in experiment dir (default: volume_denoised.npy)")
    parser.add_argument("--no-diff", action="store_true", help="Disable difference panels")
    parser.add_argument("--save", action="store_true", help="Save figure to experiments/<exp-id>/preview.png")
    parser.add_argument("--out", default=None, help="Optional output path for saved figure (overrides default)")
    args = parser.parse_args()

    phantom_path = Path("data/raw/topas_runs") / args.run_id / "phantom.npy"
    exp_dir = Path("experiments") / args.exp_id
    recon_path = exp_dir / "volume_recon.npy"
    den_path = exp_dir / args.denoised_name

    if not phantom_path.is_file():
        raise FileNotFoundError(f"Phantom not found: {phantom_path}")
    if not recon_path.is_file():
        raise FileNotFoundError(f"Reconstruction not found: {recon_path}")
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment dir not found: {exp_dir}")

    phantom = _load(phantom_path)
    recon_raw = _load(recon_path)
    recon = _align_to_phantom(phantom, recon_raw, "recon")

    has_denoised = den_path.is_file()
    denoised = None
    if has_denoised:
        den_raw = _load(den_path)
        denoised = _align_to_phantom(phantom, den_raw, "denoised")

    show_diff = not args.no_diff

    # Layout:
    # If no denoised:
    #   phantom | recon | diff(recon-gt)
    # If denoised:
    #   phantom | recon | denoised | diff(recon-gt) | diff(denoised-gt)
    if has_denoised:
        ncols = 3 + (2 if show_diff else 0)
    else:
        ncols = 2 + (1 if show_diff else 0)

    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    # Use consistent scaling for phantom/recon/denoised
    vmin = float(min(phantom.min(), recon.min(), denoised.min() if has_denoised else recon.min()))
    vmax = float(max(phantom.max(), recon.max(), denoised.max() if has_denoised else recon.max()))

    idx = 0
    im0 = _imshow(axes[idx], phantom, "Ground Truth Phantom", cmap="gray", vmin=vmin, vmax=vmax)
    plt.colorbar(im0, ax=axes[idx], fraction=0.046)
    idx += 1

    im1 = _imshow(axes[idx], recon, "MART Reconstruction", cmap="gray", vmin=vmin, vmax=vmax)
    plt.colorbar(im1, ax=axes[idx], fraction=0.046)
    idx += 1

    if has_denoised:
        im2 = _imshow(axes[idx], denoised, "Denoised Output", cmap="gray", vmin=vmin, vmax=vmax)
        plt.colorbar(im2, ax=axes[idx], fraction=0.046)
        idx += 1

    if show_diff:
        diff_r = recon - phantom
        vmax_diff_r = float(max(abs(diff_r.min()), abs(diff_r.max())))
        imd1 = _imshow(axes[idx], diff_r, "Diff: Recon − GT", cmap="bwr", vmin=-vmax_diff_r, vmax=vmax_diff_r)
        plt.colorbar(imd1, ax=axes[idx], fraction=0.046)
        idx += 1

        if has_denoised:
            diff_d = denoised - phantom
            vmax_diff_d = float(max(abs(diff_d.min()), abs(diff_d.max())))
            imd2 = _imshow(axes[idx], diff_d, "Diff: Denoised − GT", cmap="bwr", vmin=-vmax_diff_d, vmax=vmax_diff_d)
            plt.colorbar(imd2, ax=axes[idx], fraction=0.046)
            idx += 1

    fig.suptitle(f"{args.exp_id}  |  {args.run_id}", fontsize=12)
    plt.tight_layout()

    if args.save:
        out_path = Path(args.out) if args.out else (exp_dir / "preview.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved preview figure to: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()

