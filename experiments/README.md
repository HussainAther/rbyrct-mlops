# RBYRCT Experiments

This directory contains *immutable* experiment outputs.

Each experiment lives in its own subdirectory, e.g.:

- `exp_0001_mart_unet/`
- `exp_0002_sparse_angles/`
- `exp_0003_lowdose_unet/`

## Directory structure

For each experiment:

```text
experiments/exp_xxxx_name/
  config.yaml       # exact config used for this run (copied from configs/)
  metadata.json     # id, description, timestamp
  run.log           # textual log of the run
  volume_recon.npy  # MART reconstruction (1D/2D/3D ndarray)
  volume_denoised.npy  # (optional) denoised / inpainted volume
  metrics.csv       # (optional) small metrics table: ssim/psnr etc.

