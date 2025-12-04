#!/usr/bin/env python

import csv
import json
from pathlib import Path
from typing import Dict, Any, Optional, List


def load_metadata(exp_dir: Path) -> Dict[str, Any]:
    meta_path = exp_dir / "metadata.json"
    if not meta_path.is_file():
        return {}
    try:
        with meta_path.open() as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"failed to read metadata.json: {e}"}


def load_metrics(exp_dir: Path) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Returns a dict like:
    {
      "recon":   {"ssim": 0.99, "psnr": 40.5},
      "denoised":{"ssim": 0.995, "psnr": 42.1},
      ...
    }
    If metrics.csv is missing or invalid, returns {}.
    """
    metrics_path = exp_dir / "metrics.csv"
    if not metrics_path.is_file():
        return {}

    results: Dict[str, Dict[str, Optional[float]]] = {}

    try:
        with metrics_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("name") or "unknown"
                entry: Dict[str, Optional[float]] = {}
                for key in ("ssim", "psnr"):
                    val = row.get(key)
                    if val is None or val == "":
                        entry[key] = None
                    else:
                        try:
                            entry[key] = float(val)
                        except ValueError:
                            entry[key] = None
                results[name] = entry
    except Exception:
        return {}

    return results


def summarize_experiment(exp_dir: Path) -> Dict[str, Any]:
    meta = load_metadata(exp_dir)
    metrics = load_metrics(exp_dir)

    exp_id = meta.get("id", exp_dir.name)
    family = meta.get("family", "")
    variant = meta.get("variant", "")
    ts = meta.get("timestamp_utc", "")
    desc = meta.get("description", "").strip()

    recon_metrics = metrics.get("recon", {})
    denoised_metrics = metrics.get("denoised", {})

    return {
    "id": exp_id,
    "family": family,
    "variant": variant,
    "timestamp": ts,
    "description": desc,
    "recon_ssim": recon_metrics.get("ssim"),
    "recon_psnr": recon_metrics.get("psnr"),
    "denoised_ssim": denoised_metrics.get("ssim"),
    "denoised_psnr": denoised_metrics.get("psnr"),
}


def format_float(x: Optional[float]) -> str:
    if x is None:
        return "-"
    return f"{x:.4f}"


def print_table(rows: List[Dict[str, Any]]):
    if not rows:
        print("No experiments found.")
        return

    headers = [
        "family",
        "variant",
        "id",
        "timestamp",
        "recon_ssim",
        "recon_psnr",
        "denoised_ssim",
        "denoised_psnr",
    ]


    # Prepare string rows
    str_rows = []
    for r in rows:

        str_rows.append({
            "family": r["family"],
            "variant": r["variant"],
            "id": r["id"],
            "timestamp": r["timestamp"],
            "recon_ssim": format_float(r["recon_ssim"]),
            "recon_psnr": format_float(r["recon_psnr"]),
            "denoised_ssim": format_float(r["denoised_ssim"]),
            "denoised_psnr": format_float(r["denoised_psnr"]),
})

    # Compute column widths
    widths = {h: len(h) for h in headers}
    for r in str_rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(r[h])))

    # Print header
    header_line = "  ".join(h.ljust(widths[h]) for h in headers)
    sep_line = "  ".join("-" * widths[h] for h in headers)
    print(header_line)
    print(sep_line)

    # Print rows
    for r in str_rows:
        line = "  ".join(str(r[h]).ljust(widths[h]) for h in headers)
        print(line)


def main():
    base = Path("experiments")
    if not base.is_dir():
        print("No 'experiments' directory found.")
        return

    rows = []
    for exp_dir in sorted(base.iterdir()):
        if not exp_dir.is_dir():
            continue
        # Skip hidden dirs if any
        if exp_dir.name.startswith("."):
            continue
        rows.append(summarize_experiment(exp_dir))

    print_table(rows)


if __name__ == "__main__":
    main()

