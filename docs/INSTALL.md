# Installation

## Requirements

| Component | Version |
|---|---|
| OS | Linux (tested) |
| Python | 3.10 or newer |
| PyTorch / torchvision | ≥ 2.1 / ≥ 0.16 |
| Ultralytics | ≥ 8.3.0 (YOLO11 support) |
| GPU (training) | NVIDIA GPU; the paper used one RTX 5000 Ada (32 GB), CUDA 12.4, batch 16. Lower `--batch` on smaller GPUs |
| Disk | estimate: roughly 40–60 GB per processed copy of the full dataset (≈ 95 k frames, 1280×720 JPEG, quality 100); use `--quality 95` to reduce |

CPU-only machines can run evaluation, inference and the demo notebook.

## Option A — pip + venv

```bash
git clone https://github.com/SaveVic/dastrp-model.git
cd dastrp-model
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 1) PyTorch built for your CUDA version (see https://pytorch.org/get-started/locally/)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
#    CPU only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2) everything else
pip install -r requirements.txt
```

## Option B — conda

```bash
conda env create -f environment.yml    # CPU only: change pytorch-cuda=12.4 to cpuonly
conda activate dastrp
```

## Verify

```bash
python tools/check_env.py            # versions, CUDA, project imports
python tools/check_env.py --forward  # + forward pass of DASTRP-Net with random weights
```

Expected last line: `Environment OK`.

## Optional

| Purpose | Package |
|---|---|
| Demo notebook | `pip install jupyterlab` |
| Publishing weights | GitHub CLI `gh` |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.compile` errors (`train.py` compiles by default) | add `--no-compile` |
| AMP / `GradScaler` warnings on CPU | add `--no-amp` |
| Out of GPU memory | lower `--batch`, or `--workers` for host RAM |
| `weights_only` unpickling error when loading a checkpoint | use the cleaned release checkpoints from `weights/` (see `weights/README.md`) |
| Ultralytics tries to download `yolo11n.pt` | expected only when training the detector from COCO-pretrained weights |

## Versions checked for this release

The full pipeline (data conversion → detector → DASTRP-Net, ablations,
baselines → evaluation, figures, inference, notebook) was smoke-tested
on CPU with Python 3.13, PyTorch 2.14, torchvision 0.29,
Ultralytics 8.4, OpenCV 5.0, NumPy 2.5 and pandas 3.0.
