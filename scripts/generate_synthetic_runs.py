#!/usr/bin/env python

"""
Synthetic data generator for RBYRCT MART experiments.

This is a placeholder for real TOPAS-based ray-by-ray CT data.
It generates simple 1D "phantoms" and linear systems:

    projections = system_matrix @ phantom

and saves:

    - projections.npy
    - system_matrix.npy
    - geometry.json
    - phantom.npy

under:

    data/raw/topas_runs/<run_id>/

Later, you can replace this with real Monte Carlo outputs while keeping
the same file contract for the MART pipeline.
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
    n_voxels: int
    n_rays: int
    dose_factor: float  # 1.0 = baseline, < 1.0 = lower dose (more noise)
    noise_std: float    # base noise level before dose scaling
    description: str


# Pre-defined scenarios matching your roadmap
SCENARIOS = {
    # Baseline toy run (you already effectively use this via make_toy_topas_run.py)
    "topas_run_0001": Scenario(
        run_id="topas_run_0001",
        n_voxels=16,
        n_rays=16,
        dose_factor=1.0,
        noise_std=0.01,
        description="Baseline toy run, full-ish coverage, 1x dose.",
    ),

    # Sparse angles (conceptual: fewer rays, different system_matrix)
    "topas_run_sparse_90deg_0001": Scenario(
        run_id="topas_run_sparse_90deg_0001",
        n_voxels=64,
        n_rays=48,  # fewer rays than voxels
        dose_factor=1.0,
        noise_std=0.01,
        description="Conceptual 90-degree sparse-view run (fewer rays).",
    ),
    "topas_run_sparse_60deg_0001": Scenario(
        run_id="topas_run_sparse_60deg_0001",
        n_voxels=64,
        n_rays=32,
        dose_factor=1.0,
        noise_std=0.01,
        description="Conceptual 60-degree sparse-view run (even fewer rays).",
    ),

    # Low dose variants (same size, more noise)
    "topas_run_lowdose_0p5x_0001": Scenario(
        run_id="topas_run_lowdose_0p5x_0001",
        n_voxels=64,
        n_rays=64,
        dose_factor=0.5,
        noise_std=0.02,
        description="Low-dose 0.5x run, full-ish coverage, increased noise.",
    ),
    "topas_run_lowdose_0p25x_0001": Scenario(
        run_id="topas_run_lowdose_0p25x_0001",
        n_voxels=64,
        n_rays=64,
        dose_factor=0.25,
        noise_std=0.03,
        description="Low-dose 0.25x run, full-ish coverage, more noise.",
    ),
    "topas_run_lowdose_0p1x_0001": Scenario(
        run_id="topas_run_lowdose_0p1x_0001",
        n_voxels=64,
        n_rays=64,
        dose_factor=0.1,
        noise_std=0.05,
        description="Very low-dose 0.1x run, full-ish coverage, high noise.",
    ),
}


def make_phantom(n_voxels: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Simple 1D phantom: sum of a few Gaussian bumps.
    Returns a vector of shape (n_voxels,).
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n_voxels)

    # Few random Gaussian "lesions"/structures
    centers = rng.uniform(0.1, 0.9, size=3)
    widths = rng.uniform(0.05, 0.15, size=3)
    amps = rng.uniform(0.5, 1.5, size=3)

    phantom = np.zeros_like(x)
    for c, w, a in zip(centers, widths, amps):
        phantom += a * np.exp(-0.5 * ((x - c) / w) ** 2)

    # Normalize to [0, 1]
    phantom -= phantom.min()
    if phantom.max() > 0:
        phantom /= phantom.max()
    return phantom.astype(np.float32)


def make_system_matrix(n_rays: int, n_voxels: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Simple positive system matrix of shape (n_rays, n_voxels).
    Think of each row as a "ray" with non-negative weights over voxels.
    We normalize each row so that sum_j A_ij = 1 (for stability).
    """
    rng = np.random.default_rng(seed)
    A = rng.uniform(0.0, 1.0, size=(n_rays, n_voxels)).astype(np.float32)

    # Encourage some locality by smoothing along voxels
    for i in range(n_rays):
        A[i] = np.convolve(A[i], np.ones(3) / 3.0, mode="same")

    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    A /= row_sums
    return A


def simulate_projections(
    A: np.ndarray,
    phantom: np.ndarray,
    dose_factor: float,
    noise_std: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate projections = A @ phantom with additive Gaussian noise.

    dose_factor scales the noise: lower dose -> more noise.
    """
    rng = np.random.default_rng(seed)
    y_clean = A @ phantom  # shape (n_rays,)

    # Simple noise model: sigma_eff = noise_std / sqrt(dose_factor)
    dose_factor = max(dose_factor, 1e-6)
    sigma_eff = noise_std / np.sqrt(dose_factor)
    noise = rng.normal(0.0, sigma_eff, size=y_clean.shape).astype(np.float32)

    y_noisy = (y_clean + noise).astype(np.float32)
    return y_noisy


def write_run(scenario: Scenario, seed: Optional[int] = None) -> None:
    """
    Generate and save phantom, system_matrix, projections, and geometry.json
    for a given scenario.
    """
    run_dir = BASE_DIR / scenario.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Generating {scenario.run_id} ===")
    print(f"  n_voxels    = {scenario.n_voxels}")
    print(f"  n_rays      = {scenario.n_rays}")
    print(f"  dose_factor = {scenario.dose_factor}")
    print(f"  description = {scenario.description}")

    phantom = make_phantom(scenario.n_voxels, seed=seed)
    A = make_system_matrix(scenario.n_rays, scenario.n_voxels, seed=seed)
    projections = simulate_projections(
        A,
        phantom,
        dose_factor=scenario.dose_factor,
        noise_std=scenario.noise_std,
        seed=seed,
    )

    # Save arrays
    np.save(run_dir / "phantom.npy", phantom)
    np.save(run_dir / "system_matrix.npy", A)
    np.save(run_dir / "projections.npy", projections)

    # Save a lightweight "geometry" placeholder
    geom = {
        "description": scenario.description,
        "n_voxels": int(scenario.n_voxels),
        "n_rays": int(scenario.n_rays),
        "dose_factor": float(scenario.dose_factor),
        "note": "Synthetic 1D linear model; not physical CT geometry. "
                "Replace with real RBYRCT/TOPAS outputs later.",
    }
    with (run_dir / "geometry.json").open("w") as f:
        json.dump(geom, f, indent=2)

    print(f"  Saved phantom.npy, system_matrix.npy, projections.npy, geometry.json in {run_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic 'TOPAS-like' runs for RBYRCT MART pipeline."
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

