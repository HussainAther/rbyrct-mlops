use std::fs::File;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use ndarray::{Array1, Array2};
use ndarray_npy::{NpzReader, write_npy};

use recon_core::mart_reconstruct;

/// Simple MART CLI for RBYRCT.
///
/// Expected NPZ file structure:
///   - key "projections": 1D array (M,) of f32
///   - key "system_matrix": 2D array (M, N) of f32
///
/// Geometry JSON is currently only checked for existence. In the future,
/// it can be parsed to construct the system matrix from ray geometry.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to NPZ file containing projections and system_matrix
    #[arg(long)]
    projections: PathBuf,

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

    // --- Load projections + system matrix from NPZ ---
    let file = File::open(&args.projections)
        .map_err(|e| anyhow::anyhow!("Failed to open NPZ {:?}: {}", args.projections, e))?;
    let mut npz = NpzReader::new(file)
        .map_err(|e| anyhow::anyhow!("Failed to read NPZ {:?}: {}", args.projections, e))?;

    let projections: Array1<f32> = npz
        .by_name("projections")
        .map_err(|e| anyhow::anyhow!("Missing or invalid 'projections' array in NPZ: {}", e))?;

    let system_matrix: Array2<f32> = npz
        .by_name("system_matrix")
        .map_err(|e| anyhow::anyhow!("Missing or invalid 'system_matrix' array in NPZ: {}", e))?;

    // --- Check geometry file exists (not yet used) ---
    let _geom_file = File::open(&args.geometry)
        .map_err(|e| anyhow::anyhow!("Failed to open geometry JSON {:?}: {}", args.geometry, e))?;
    // In the future: parse geometry here and verify consistency.

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

