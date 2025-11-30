use std::fs::File;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use ndarray::{Array1, Array2};
use ndarray_npy::{read_npy, write_npy};

use recon_core::mart_reconstruct;

/// Simple MART CLI for RBYRCT.
///
/// Expects:
///   --projections: path to projections.npy (1D array, length M)
///   --system-matrix: path to system_matrix.npy (2D array, shape (M, N))
///   --geometry: path to geometry.json (currently unused, just validated)
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to projections .npy file (shape (M,))
    #[arg(long)]
    projections: PathBuf,

    /// Path to system matrix .npy file (shape (M, N))
    #[arg(long = "system-matrix")]
    system_matrix: PathBuf,

    /// Path to geometry JSON (currently unused, just validated)
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

fn main() -> Result<()> {
    let args = Args::parse();

    // --- Load projections + system matrix from .npy files ---
    let projections: Array1<f32> = read_npy(&args.projections)
        .map_err(|e| anyhow::anyhow!("Failed to read projections NPY {:?}: {}", args.projections, e))?;

    let system_matrix: Array2<f32> = read_npy(&args.system_matrix)
        .map_err(|e| anyhow::anyhow!("Failed to read system matrix NPY {:?}: {}", args.system_matrix, e))?;

    // --- Check geometry file exists (not yet used) ---
    let _geom_file = File::open(&args.geometry)
        .map_err(|e| anyhow::anyhow!("Failed to open geometry JSON {:?}: {}", args.geometry, e))?;
    // Future: parse geometry and verify consistency.

    println!(
        "Running MART with M = {}, N = {}, n_iters = {}, relaxation = {}",
        system_matrix.dim().0,
        system_matrix.dim().1,
        args.n_iters,
        args.relaxation
    );

    // --- Run MART reconstruction ---
    let volume = mart_reconstruct(&projections, &system_matrix, args.n_iters, args.relaxation);

    // --- Save volume as .npy ---
    write_npy(&args.output, &volume)
        .map_err(|e| anyhow::anyhow!("Failed to write output NPY {:?}: {}", args.output, e))?;

    println!("Reconstruction written to {:?}", args.output);

    Ok(())
}

