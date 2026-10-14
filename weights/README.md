# Pre-trained weights

Weight files are **not stored in git**. They are attached to the GitHub
Release of this repository; `manifest.json` (committed) lists every file
with its size and SHA-256.

## Download

```bash
python tools/download_weights.py                 # everything in manifest.json
python tools/download_weights.py --only dastrp_exp3.pth yolo11n_motor.pt dastrp_exp3_config.json
```

| File | Content |
|---|---|
| `yolo11n_motor.pt` | YOLO11n motorcycle detector (frozen backbone + detector of DASTRP-Net) |
| `dastrp_exp3.pth` | DASTRP-Net, Exp3 (main model in the paper) |
| `dastrp_exp3_config.json` | training configuration of that model |
| `dastrp_exp3_result_test*.json` | test-set metrics reported in the paper |
| `ablation_<name>.pth`, `baseline_b{1,2,3}.pth` | optional: ablation / baseline models |

All checkpoints contain only `model_state`, the run configuration (with
repository-relative paths), `epoch` and `val_acc`; no optimizer state and
no machine-specific paths.

## Usage

```bash
# evaluate on the test split
python scripts/evaluate.py \
    --config     weights/dastrp_exp3_config.json \
    --checkpoint weights/dastrp_exp3.pth \
    --yolo       weights/yolo11n_motor.pt \
    --dataset    data/processed/dastrp \
    --out-dir    runs/eval_release

# run on a video
python scripts/inference.py \
    --checkpoint weights/dastrp_exp3.pth \
    --yolo       weights/yolo11n_motor.pt \
    --input      data/imbd-esr/videos/ugm/ugm_1_1.mp4 \
    --output     runs/demo/ugm_1_1_annotated.mp4
```

See also `notebooks/demo_inference.ipynb`.

## Publishing (maintainers)

```bash
python tools/export_checkpoint.py --yolo runs/yolo/v11n/weights/best.pt \
       --run runs/exp3 --name dastrp_exp3 --out weights --tag v1.0.0
# optional
python tools/export_checkpoint.py --run runs/baseline/b2 --name baseline_b2 --out weights

python tools/check_release.py              # must report no findings
git add weights/manifest.json && git commit -m "Release weights v1.0.0"
gh release create v1.0.0 weights/*.pt weights/*.pth weights/*_config.json \
       weights/*_result_test*.json --title "DASTRP-Net v1.0.0"
```

`export_checkpoint.py` strips optimizer/scaler state, rewrites all
absolute paths and aborts if any absolute path is still found inside an
exported file. Never upload `runs/**/best_model.pth` directly: raw
training checkpoints contain the local paths of the training machine.
