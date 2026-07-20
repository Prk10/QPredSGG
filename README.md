# CFEN Baseline & Quantum (PyTorch)

This repository provides a **reproducible baseline** and a faithful implementation of the **Causal Features Enhancement Network (CFEN)** components for Scene Graph Generation (SGG), featuring both classical and quantum variants. It focuses on:
- A clean baseline relation head using precomputed object features (with an alternative quantum variant)
- CFEN **fact** and **counterfactual** branches
- **Class-generic** feature moving averages (EMA)
- **DM Loss** and fusion of logits
- Training and evaluation scaffolding for both classical and quantum pipelines
- A **synthetic dataset option** so you can verify your pipeline end-to-end without downloading VG

> This code is designed to be *practical* for reproduction. You can later plug in real precomputed features (e.g., from VG150) as `.npz` files per image.

## Quickstart (with synthetic data)
```bash
# Create and activate an environment (example with conda)
conda create -n cfen python=3.10 -y
conda activate cfen

# Install deps
pip install -r requirements.txt

# Train classical baseline with synthetic data (tiny demo run)
python train.py --config configs/default.yaml

# Train quantum variant with synthetic data (tiny demo run)
python train.py --config configs/quantum.yaml