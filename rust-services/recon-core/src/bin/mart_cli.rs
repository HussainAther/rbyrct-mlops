use std::fs::File;
use std::path::PathBuf;

use clap::Parser;
use ndarray::{Array1, Array2};
use ndarray_npy::NpzReader;

use recon_core::mart_reconstruct;

/// Simple MART CLI for RBYRCT.
/// 
/// Expected NPZ file structure:
///   projections.npz contains:
///     - "projections": 1D array (M,) of f32
///     - "system_matrix": 2D array (M, N) of f32
///
/// Geometry JSON is currently loaded but not used (reserved for future
/// RBYRCT steering / ray-path configuration).
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to NPZ file containing "projections" and "system_matrix"
    #[arg(long)]
    projections: PathBuf,

    /// Path to geometry JSON (not yet used, but reserved)
    #[arg(long)]
    geometry: PathBuf,

    /// Number of MART iterations
    #[arg(long, default_value_t = 50)]
    n_iters: usize,

    /// Relaxation parameter
    #[arg(long, default_value_t = 0.5)]
    relaxation: f32,

    /// Output path for reconstructed volume (.npy)
    #[arg(long)]
    output: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // --- Load projections + system matrix from NPZ ---
    let file = File::open(&args.projections)
        .map_err(|e| anyhow::anyhow!("Failed to open projections NPZ {:?}: {}", args.projections, e))?;
    let mut npz = NpzReader::new(file)
        .map_err(|e| anyhow::anyhow!("Failed to read NPZ {:?}: {}", args.projections, e))?;

    let projections: Array1<f32> = npz
        .by_name("projections")
        .map_err(|e| anyhow::anyhow!("Missing or invalid 'projections' array in NPZ: {}", e))?;

    let system_matrix: Array2<f32> = npz
        .by_name("system_matrix")
        .map_err(|e| anyhow::anyhow!("Missing or invalid 'system_matrix' array in NPZ: {}", e))?;

    // --- Load geometry JSON (currently unused, but we validate it exists) ---
    let _geom_file = File::open(&args.geometry)
        .map_err(|e| anyhow::anyhow!("Failed to open geometry JSON {:?}: {}", args.geometry, e))?;
    // If you want to parse it:
    // let geom: serde_json::Value = serde_json::from_reader(_geom_file)?;
    // For now, we just ensure it exists and is readable.

    println!(
        "Running MART with M={} rays, N={} voxels, n_iters={}, relaxation={}",
        system_matrix.dim().0,
        system_matrix.dim().1,
        args.n_iters,
        args.relaxation
    );

    // --- Run MART reconstruction ---
    let volume = mart_reconstruct(&projections, &system_matrix, args.n_iters, args.relaxation);

    // --- Save volume as .npy ---
    ndarray_npy::write_npy(&args.output, &volume)
        .map_err(|e| anyhow::anyhow!("Failed to write output NPY {:?}: {}", args.output, e))?;

    println!("Reconstruction written to {:?}", args.output);

    Ok(())
}
```

### Notes:

* This uses `anyhow` for nicer errors, so add it to `Cargo.toml`:

  ```toml
  anyhow = "1.0"
  ```

* Expected **NPZ contents** (your Python side must create this):

  ```python
  import numpy as np

  # Example: M rays, N voxels
  projections = np.array([...], dtype=np.float32)      # shape (M,)
  system_matrix = np.array([...], dtype=np.float32)    # shape (M, N)

  np.savez("data/raw/topas_runs/topas_run_0001/projections.npz",
           projections=projections,
           system_matrix=system_matrix)
  ```

  Then your config:

  ```yaml
  sim:
    data_path: data/raw/topas_runs/topas_run_0001
    projections_file: projections.npz
    geometry_file: geometry.json    # can be an empty {} for now
