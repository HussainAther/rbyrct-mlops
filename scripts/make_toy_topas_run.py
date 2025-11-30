# scripts/make_toy_topas_run.py

import json
from pathlib import Path

import numpy as np


def main():
    base = Path("data/raw/topas_runs/topas_run_0001")
    base.mkdir(parents=True, exist_ok=True)

    # ---- Toy system: 2 rays, 2 voxels ----
    system_matrix = np.array(
        [
            [1.0, 0.0],  # ray 0 -> voxel 0
            [0.0, 1.0],  # ray 1 -> voxel 1
        ],
        dtype=np.float32,
    )

    phantom = np.array([1.0, 0.5], dtype=np.float32)  # ground truth (2 voxels)

    projections = system_matrix @ phantom  # shape (2,)

    # Save as separate .npy files
    proj_path = base / "projections.npy"
    sysm_path = base / "system_matrix.npy"
    np.save(proj_path, projections)
    np.save(sysm_path, system_matrix)
    print(f"Saved projections to {proj_path}")
    print(f"Saved system matrix to {sysm_path}")

    # Geometry JSON placeholder
    geom = {
        "description": "toy 2-ray, 2-voxel system",
        "num_rays": int(system_matrix.shape[0]),
        "num_voxels": int(system_matrix.shape[1]),
        "note": "placeholder geometry; real RBYRCT geometry comes later.",
    }
    geom_path = base / "geometry.json"
    with geom_path.open("w") as f:
        json.dump(geom, f, indent=2)
    print(f"Saved geometry JSON to {geom_path}")

    # Save phantom for metrics
    phantom_path = base / "phantom.npy"
    np.save(phantom_path, phantom)
    print(f"Saved phantom (ground truth) to {phantom_path}")


if __name__ == "__main__":
    main()

