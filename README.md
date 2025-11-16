# 🌀 RBYRCT MLOps — Rust + AI Pipeline for Ray-by-Ray CT

**Goal:**  
Practice end-to-end MLOps using **RBYRCT** (Ray-by-Ray Computed Tomography) as a realistic playground:

- Rust services for **high-performance reconstruction**
- Python for **ML training & experiments**
- Reproducible configs, data, and deployment for **scientific + production** workflows

This repo is intentionally overkill in a *good* way — it’s a sandbox for building real skills.

---

## 🧠 High-Level Architecture

**Pipeline idea:**

1. **Simulation / Ingest**  
   - Projections & geometry from TOPAS, Houdini, or synthetic generators.
   - Optional `ingest` Rust service to receive and store projections.

2. **Reconstruction (Rust)**  
   - `recon-core`: MART/SART/other iterative methods implemented in Rust.  
   - `recon-api`: HTTP/gRPC service using `recon-core` to reconstruct volumes.

3. **AI Denoising / Post-Processing (Python → Rust)**  
   - Python training (`training/rbyrct_denoiser`) → export ONNX / TorchScript.  
   - `ai-infer` Rust service loads model and performs fast inference.

4. **Orchestration & Ops (optional layers)**  
   - `scheduler` service to kick off recon + AI jobs.  
   - `ops/` contains Docker, k8s, and CI scaffolding.

The idea: you can call one endpoint with projections and get back a reconstructed (and optionally denoised) volume or slice stack.

---

## 📂 Repo Structure

```text
rbyrct-mlops/
  README.md
  configs/           # YAML configs for experiments
  data/              # raw + processed (DVC/LFS recommended)
  notebooks/         # exploration, prototyping
  training/          # Python-based ML training
  rust-services/     # all Rust crates (workspace)
  ops/               # deployment, CI, infra
````

### Rust Services (`rust-services/`)

* `recon-core/`
  Core math and reconstruction algorithms:

  * Grid/voxel structures
  * Projection / backprojection
  * MART/SART iterations
  * Geometry helpers (angles, SAD, SDD)

* `recon-api/`
  HTTP/gRPC API around `recon-core`:

  * `POST /reconstruct`
    Body: projections + geometry
    Response: volume ID, quick metrics, maybe a preview.

* `ai-infer/`
  Wraps an ONNX/TorchScript model for:

  * Denoising
  * Super-resolution
  * Artifact reduction

* (Optional) `ingest/`, `scheduler/`
  For more advanced event-driven pipelines.

### Python (`training/`)

* `datasets.py`:

  * Loads projections & phantoms from `data/`
  * Augmentations (noise, random angles, dose levels)

* `models.py`:

  * Simple UNet / DnCNN / transformer-based denoiser

* `train.py`:

  * Train + log metrics (SSIM, PSNR, etc.)
  * Save checkpoints and export ONNX/TorchScript

* `infer.py`:

  * CLI/utility to run inference on stored reconstructions

---

## 🚀 Getting Started

### 1. Clone & basic setup

```bash
git clone https://github.com/<your-username>/rbyrct-mlops.git
cd rbyrct-mlops
```

### 2. Rust toolchain

Install Rust (if you haven’t):

```bash
curl https://sh.rustup.rs -sSf | sh
```

Inside the project:

```bash
cd rust-services
cargo build --workspace
```

### 3. Python environment (for training & experiments)

Using `conda` (recommended):

```bash
conda create -n rbyrct-mlops python=3.11 -y
conda activate rbyrct-mlops

pip install -r requirements.txt
# or, if using pyproject.toml:
# pip install .
```

Suggested core deps:

* `torch`, `torchvision`
* `numpy`, `scipy`
* `matplotlib`
* `tqdm`
* `onnx`, `onnxruntime`
* `mlflow` or `wandb` (optional)

---

## 🧪 Running a Minimal E2E Flow

> This is the “hello world” pipeline: projections → Rust recon → Python denoise.

1. **Prepare a tiny synthetic dataset**

You can start by putting a small test file under `data/raw/`:

```bash
data/
  raw/
    example_projections.npz      # projections + geometry
```

You’ll later replace this with real TOPAS/Houdini-exports.

2. **Run reconstruction via Rust service**

In one terminal:

```bash
cd rust-services/recon-api
cargo run
```

In another terminal (simple HTTP example, assuming axum):

```bash
curl -X POST http://localhost:8080/reconstruct \
  -H "Content-Type: application/json" \
  -d @configs/example_lowdose.json
```

Expected response (conceptually):

```json
{
  "volume_id": "vol_123abc",
  "ssim_estimate": 0.88
}
```

3. **Denoise / refine with Python**

```bash
python -m training.rbyrct_denoiser.infer \
  --volume-id vol_123abc \
  --output-path data/processed/vol_123abc_denoised.npz
```

Later, `ai-infer` will let you do this straight in Rust.

---

## 🧮 MART / Reconstruction Core (Concept)

MART loop in Rust (sketch):

```rust
// rust-services/recon-core/src/mart.rs

pub fn mart_step(
    projections: &Array2<f32>,
    system_matrix: &Array2<f32>,
    volume: &mut Array1<f32>,
    relaxation: f32,
) {
    // For each ray, compute ratio measured / estimated
    // and update the volume multiplicatively.
}
```

Over time, you can:

* Move from toy 2D to real 3D volumes
* Explore GPU-accelerated kernels
* Integrate with real simulation geometry

---

## 📈 MLOps & Ops Hooks

This repo is a playground to explore:

* **Artifact tracking** (models, volumes, metrics)
* **Data versioning** (DVC, Git LFS)
* **CI** (formatting, tests, small “smoke” recon job)
* **Containerization** (Dockerfiles under `ops/docker`)
* **Deployment** (k8s manifests under `ops/k8s`)

The idea isn’t to “boil the ocean” but to gradually add pieces as you experiment:

* Step 1: local CLIs
* Step 2: one Rust API + one Python trainer
* Step 3: messaging/orchestration + monitoring

---

## 🧭 Roadmap (Personal Practice)

Some ideas for practice milestones:

* [ ] Implement 2D MART in `recon-core` and unit test against a Python reference.
* [ ] Build a simple `POST /reconstruct` in `recon-api` that returns a PNG slice.
* [ ] Train a UNet denoiser in `training/` and export to ONNX.
* [ ] Integrate ONNX inference into `ai-infer` and benchmark latency.
* [ ] Add basic metrics (SSIM/PSNR) and log them per run.
* [ ] Dockerize `recon-api` and run it locally.
* [ ] Optional: deploy the whole pipeline in a local k8s cluster (kind/minikube).

You don’t have to do all of this at once. This repo is meant to be your **practice arena** while you build RBYRCT and your systems-engineering brain in parallel.

---

## 🪪 License

Pick what fits your goals, e.g.:

* Code: MIT or Apache-2.0
* Data: CC BY 4.0 (or stricter if needed)

---

## 🙏 Acknowledgments

This project is inspired by:

* The **Ray-by-Ray Computed Tomography (RBYRCT)** and steerable X-ray concept.
* Work on low-dose imaging, iterative reconstruction, and ML-based denoising.
* The Rust, Python, and open-source communities that make this kind of pipeline possible.

---

```

---

If you want, next step I can:

- Turn the **Rust workspace** part into a ready-to-copy `Cargo.toml`  
- Or sketch a **tiny first implementation** of `recon-core` with a fake MART loop + unit test so you can paste it straight in and `cargo test` it.
::contentReference[oaicite:0]{index=0}
```

