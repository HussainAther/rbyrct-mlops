#!/usr/bin/env python

"""
Synthetic data generator for RBYRCT MART experiments (2D phantom version).

This generates simple 2D "phantoms" and linear systems:

    x      : flattened 2D phantom of shape (ny * nx,)
    A      : system matrix of shape (n_rays, ny * nx)
    y      : projections = A @ x

and saves:

    - phantom.npy        # shape (ny, nx)    (2D image)
    - system_matrix.npy  # shape (n_rays, ny * nx)
    - projections.npy    # shape (n_rays,)
    - geometry.json

under:

    data/raw/topas_runs/<run_id>/

Later, you can replace this with real TOPAS-based RBYRCT outputs while
keeping the same on-disk contract for the MART pipeline.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


BASE_DIR = Path("data/raw/topas_runs")


@dataclass
class Scenario:
    run_id: str
    nx: int          # image width
    ny: int          # image height
    n_rays: int      # number of "rays" (rows of A)
    dose_factor: float  # 1.0 = baseline, < 1.0 = lower dose (more noise)
    noise_std: float    # base noise level before dose scaling
    description: str


# Pre-defined scenarios matching your roadmap (now 2D)
SCENARIOS = {
    # Baseline toy run
    "topas_run_0001": Scenario(
        run_id="topas_run_0001",
        nx=16,
        ny=16,
        n_rays=64,
        dose_factor=1.0,
        noise_std=0.01,
        description="Baseline toy run, 16x16 phantom, moderate number of rays, 1x dose.",
    ),

    # Sparse angles (conceptual: fewer rays)
    "topas_run_sparse_90deg_0001": Scenario(
        run_id="topas_run_sparse_90deg_0001",
        nx=32,
        ny=32,
        n_rays=96,  # fewer rays than full-coverage case
        dose_factor=1.0,
        noise_std=0.01,
        description="Conceptual 90-degree sparse-view run on 32x32 phantom.",
    ),
    "topas_run_sparse_60deg_0001": Scenario(
        run_id="topas_run_sparse_60deg_0001",
        nx=32,
        ny=32,
        n_rays=64,  # even fewer rays
        dose_factor=1.0,
        noise_std=0.01,
        description="Conceptual 60-degree sparse-view run on 32x32 phantom.",
    ),

    # Low-dose variants (same size, more noise)
    "topas_run_lowdose_0p5x_0001": Scenario(
        run_id="topas_run_lowdose_0p5x_0001",
        nx=32,
        ny=32,
        n_rays=128,
        dose_factor=0.5,
        noise_std=0.02,
        description="Low-dose 0.5x run on 32x32 phantom, increased noise.",
    ),
    "topas_run_lowdose_0p25x_0001": Scenario(
        run_id="topas_run_lowdose_0p25x_0001",
        nx=32,
        ny=32,
        n_rays=128,
        dose_factor=0.25,
        noise_std=0.03,
        description="Low-dose 0.25x run on 32x32 phantom, more noise.",
    ),
    "topas_run_lowdose_0p1x_0001": Scenario(
        run_id="topas_run_lowdose_0p1x_0001",
        nx=32,
        ny=32,
        n_rays=128,
        dose_factor=0.1,
        noise_std=0.05,
        description="Very low-dose 0.1x run on 32x32 phantom, high noise.",
    ),
}


def make_phantom_2d(nx: int, ny: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Simple 2D phantom: sum of Gaussian blobs (like tiny lesions / structures).
    Returns an array of shape (ny, nx).
    """
    rng = np.random.default_rng(seed)

    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(xs, ys)

    phantom = np.zeros((ny, nx), dtype=np.float32)

    # A few random Gaussian blobs
    n_blobs = 4
    centers_x = rng.uniform(0.2, 0.8, size=n_blobs)
    centers_y = rng.uniform(0.2, 0.8, size=n_blobs)
    widths = rng.uniform(0.05, 0.15, size=n_blobs)
    amps = rng.uniform(0.5, 1.5, size=n_blobs)

    for cx, cy, w, a in zip(centers_x, centers_y, widths, amps):
        phantom += a * np.exp(-0.5 * (((X - cx) / w) ** 2 + ((Y - cy) / w) ** 2))

    # Normalize to [0, 1]
    phantom -= phantom.min()
    max_val = phantom.max()
    if max_val > 0:
        phantom /= max_val

    return phantom.astype(np.float32)


def make_system_matrix(n_rays: int, n_voxels: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Simple positive system matrix of shape (n_rays, n_voxels).
    Each row is a "ray" with non-negative weights over voxels.
    We normalize each row so that sum_j A_ij = 1 (for stability with MART).
    """
    rng = np.random.default_rng(seed)
    A = rng.uniform(0.0, 1.0, size=(n_rays, n_voxels)).astype(np.float32)

    # Mild smoothing along voxel dimension to encourage locality
    # (this is 1D over flattened voxels; not a real CT geometry).
    kernel = np.ones(5, dtype=np.float32) / 5.0
    for i in range(n_rays):
        A[i] = np.convolve(A[i], kernel, mode="same")

    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    A /= row_sums
    return A


def simulate_projections(
    A: np.ndarray,
    phantom_flat: np.ndarray,
    dose_factor: float,
    noise_std: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate projections = A @ phantom_flat with additive Gaussian noise.

    dose_factor scales the noise: lower dose -> more effective noise.
    """
    rng = np.random.default_rng(seed)
    y_clean = A @ phantom_flat  # shape (n_rays,)

    dose_factor = max(dose_factor, 1e-6)
    sigma_eff = noise_std / np.sqrt(dose_factor)
    noise = rng.normal(0.0, sigma_eff, size=y_clean.shape).astype(np.float32)

    y_noisy = (y_clean + noise).astype(np.float32)
    return y_noisy


def write_run(scenario: Scenario, seed: Optional[int] = None) -> None:
    """
    Generate and save phantom (2D), system_matrix, projections, and geometry.json
    for a given scenario.
    """
    run_dir = BASE_DIR / scenario.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    n_voxels = scenario.nx * scenario.ny

    print(f"=== Generating {scenario.run_id} ===")
    print(f"  image size  = {scenario.ny} x {scenario.nx}")
    print(f"  n_voxels    = {n_voxels}")
    print(f"  n_rays      = {scenario.n_rays}")
    print(f"  dose_factor = {scenario.dose_factor}")
    print(f"  description = {scenario.description}")

    phantom_2d = make_phantom_2d(scenario.nx, scenario.ny, seed=seed)
    phantom_flat = phantom_2d.reshape(-1)  # (n_voxels,)

    A = make_system_matrix(scenario.n_rays, n_voxels, seed=seed)
    projections = simulate_projections(
        A,
        phantom_flat,
        dose_factor=scenario.dose_factor,
        noise_std=scenario.noise_std,
        seed=seed,
    )

    # Save arrays
    np.save(run_dir / "phantom.npy", phantom_2d)         # shape (ny, nx)
    np.save(run_dir / "system_matrix.npy", A)            # shape (n_rays, n_voxels)
    np.save(run_dir / "projections.npy", projections)    # shape (n_rays,)

    # Save a lightweight "geometry" placeholder
    geom = {
        "description": scenario.description,
        "nx": int(scenario.nx),
        "ny": int(scenario.ny),
        "n_voxels": int(n_voxels),
        "n_rays": int(scenario.n_rays),
        "dose_factor": float(scenario.dose_factor),
        "note": "Synthetic 2D linear model; not physical CT geometry. "
                "Replace with real RBYRCT/TOPAS outputs later.",
    }
    with (run_dir / "geometry.json").open("w") as f:
        json.dump(geom, f, indent=2)

    print(f"  Saved phantom.npy (2D), system_matrix.npy, projections.npy, geometry.json in {run_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic 2D 'TOPAS-like' runs for RBYRCT MART pipeline."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        choices=sorted(SCENARIOS.keys()),
        help="Which predefined run_id to generate.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all predefined scenarios.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    if not args.all and not args.run_id:
        parser.error("Either --run-id or --all must be specified.")

    if args.all:
        for sc in SCENARIOS.values():
            write_run(sc, seed=args.seed)
    else:
        sc = SCENARIOS[args.run_id]
        write_run(sc, seed=args.seed)


if __name__ == "__main__":
    main()

