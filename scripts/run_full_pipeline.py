# scripts/run_full_pipeline.py

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

# Optional: if you want SSIM/PSNR, install scikit-image:
# pip install scikit-image
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def log(msg: str, log_path: Optional[Path] = None):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp} UTC] {msg}"
    print(line)
    if log_path is not None:
        with log_path.open("a") as f:
            f.write(line + "\n")


def load_config(cfg_path: Path) -> dict:
    with cfg_path.open() as f:
        return yaml.safe_load(f)


def prepare_experiment_dir(cfg: dict) -> Path:
    base_dir = Path(cfg["output"]["base_dir"])
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def run_rust_mart(cfg: dict, exp_dir: Path, log_path: Path) -> Path:
    recon_cfg = cfg["recon"]
    sim_cfg = cfg["sim"]

    rust_bin = Path(recon_cfg["rust_binary"])
    if not rust_bin.exists():
        log("[WARN] Rust binary not found at {} (you may need to build it).".format(rust_bin), log_path)

    projections_path = Path(sim_cfg["data_path"]) / sim_cfg["projections_file"]
    system_matrix_path = Path(sim_cfg["data_path"]) / sim_cfg["system_matrix_file"]
    geometry_path = Path(sim_cfg["data_path"]) / sim_cfg["geometry_file"]
    recon_output = exp_dir / "volume_recon.npy"

    cmd = [
        str(rust_bin),
        "--projections", str(projections_path),
        "--system-matrix", str(system_matrix_path),
        "--geometry", str(geometry_path),
        "--n-iters", str(recon_cfg["n_iters"]),
        "--relaxation", str(recon_cfg["relaxation"]),
        "--output", str(recon_output),
    ]



    log("Running Rust MART: " + " ".join(cmd), log_path)
    subprocess.run(cmd, check=True)
    log("Rust MART finished. Recon saved to {}".format(recon_output), log_path)

    return recon_output


def load_model(model_cfg: dict):
    module_name = model_cfg["module"]
    class_name = model_cfg["class_name"]
    checkpoint = model_cfg["checkpoint"]
    device = model_cfg.get("device", "auto")

    module = import_module(module_name)
    cls = getattr(module, class_name)

    import torch

    model = cls()

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, device


def run_denoiser(cfg: dict, recon_path: Path, exp_dir: Path, log_path: Path) -> Path:
    model_cfg = cfg["model"]
    output_path = exp_dir / "volume_denoised.npy"

    log("Loading denoiser model...", log_path)
    model, device = load_model(model_cfg)

    import torch

    vol = np.load(recon_path)
    # Handle only 2D for now
    if vol.ndim == 2:
        x = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)
    elif vol.ndim == 3:
        mid = vol.shape[0] // 2
        x = torch.from_numpy(vol[mid]).float().unsqueeze(0).unsqueeze(0)
        log("[WARN] 3D volume detected; using middle slice only.", log_path)
    else:
        raise ValueError("Unexpected volume shape: {}".format(vol.shape))

    x = x.to(device)
    with torch.no_grad():
        y = model(x).cpu().squeeze().numpy()

    np.save(output_path, y)
    log("Denoised volume saved to {}".format(output_path), log_path)

    return output_path


def compute_metrics(
    cfg: dict,
    recon_path: Path,
    denoised_path: Optional[Path],
    exp_dir: Path,
    log_path: Path
):
    metrics_cfg = cfg.get("metrics", {})
    compute_ssim_flag = metrics_cfg.get("compute_ssim", False)
    compute_psnr_flag = metrics_cfg.get("compute_psnr", False)

    phantom_path = metrics_cfg.get("gt_volume_path") or cfg["sim"].get("phantom_file")
    if phantom_path is None:
        log("No ground truth phantom; skipping metrics.", log_path)
        return

    phantom_path = Path(phantom_path)
    if not phantom_path.is_file():
        alt = Path(cfg["sim"]["data_path"]) / phantom_path
        if alt.is_file():
            phantom_path = alt
        else:
            log("Phantom not found; skipping metrics.", log_path)
            return

    if not HAS_SKIMAGE and (compute_ssim_flag or compute_psnr_flag):
        log("scikit-image not installed; skipping metrics.", log_path)
        return

    import csv
    gt = np.load(phantom_path)
    recon = np.load(recon_path)

    metrics_rows = []

    entry = {"name": "recon"}
    if compute_ssim_flag:
        entry["ssim"] = float(ssim(gt, recon, data_range=gt.max() - gt.min()))
    if compute_psnr_flag:
        entry["psnr"] = float(psnr(gt, recon, data_range=gt.max() - gt.min()))
    metrics_rows.append(entry)

    if denoised_path is not None and denoised_path.is_file():
        den = np.load(denoised_path)
        entry = {"name": "denoised"}
        if compute_ssim_flag:
            entry["ssim"] = float(ssim(gt, den, data_range=gt.max() - gt.min()))
        if compute_psnr_flag:
            entry["psnr"] = float(psnr(gt, den, data_range=gt.max() - gt.min()))
        metrics_rows.append(entry)

    metrics_file = exp_dir / cfg["output"].get("metrics_file", "metrics.csv")
    with metrics_file.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "ssim", "psnr"])
        writer.writeheader()
        for row in metrics_rows:
            writer.writerow(row)

    log("Metrics written to {}".format(metrics_file), log_path)


def save_metadata(cfg: dict, exp_dir: Path, cfg_path: Path, log_path: Path):
    if cfg["output"].get("save_config_copy", True):
        dst = exp_dir / "config.yaml"
        shutil.copy(cfg_path, dst)
        log("Config copied to {}".format(dst), log_path)

    meta = {
        "id": cfg["id"],
        "description": cfg.get("description", ""),
        "timestamp_utc": datetime.utcnow().isoformat(),
    }
    meta_path = exp_dir / "metadata.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    log("Metadata saved to {}".format(meta_path), log_path)


def main():
    parser = argparse.ArgumentParser(description="Run full RBYRCT MART + denoiser pipeline.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    exp_dir = prepare_experiment_dir(cfg)
    log_path = exp_dir / cfg["output"].get("log_file", "run.log")

    log("Starting experiment {}".format(cfg["id"]), log_path)

    recon_path = run_rust_mart(cfg, exp_dir, log_path)

    denoised_path = None
    if cfg.get("model", {}).get("checkpoint"):
        denoised_path = run_denoiser(cfg, recon_path, exp_dir, log_path)
    else:
        log("No checkpoint; skipping denoising.", log_path)

    compute_metrics(cfg, recon_path, denoised_path, exp_dir, log_path)
    save_metadata(cfg, exp_dir, cfg_path, log_path)

    log("Experiment {} completed.".format(cfg["id"]), log_path)


if __name__ == "__main__":
    main()

