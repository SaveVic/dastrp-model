"""
tools/check_env.py
==================
Verify that the environment can run DASTRP-Net.

  python tools/check_env.py            # versions, CUDA, project imports
  python tools/check_env.py --forward  # + tiny forward pass with a random YOLO11n
"""

import argparse
import importlib
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REQUIRED = {
    "torch": "2.1", "torchvision": "0.16", "ultralytics": "8.3.0",
    "cv2": "4.8", "numpy": "1.24", "tqdm": "4.65",
    "tensorboard": "2.14", "matplotlib": "3.7", "pandas": "2.0",
}


def _ver(v: str) -> tuple:
    out = []
    for p in v.split("+")[0].split("."):
        digits = "".join(ch for ch in p if ch.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forward", action="store_true")
    args = ap.parse_args()

    ok = True
    print(f"Python      {platform.python_version()}  ({platform.system()})")
    if sys.version_info < (3, 10):
        print("  ✗ Python >= 3.10 required"); ok = False

    for mod, minimum in REQUIRED.items():
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "?")
            flag = "✓" if _ver(v) >= _ver(minimum) else f"✗ (need >= {minimum})"
            ok &= flag == "✓"
            print(f"{mod:<12}{v:<14}{flag}")
        except ImportError:
            print(f"{mod:<12}{'missing':<14}✗"); ok = False

    import torch
    if torch.cuda.is_available():
        print(f"CUDA        {torch.version.cuda}  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA        not available (CPU mode: use --no-amp, expect slow training)")

    for mod in ("models.architecture", "models.baseline_b1", "data.dataset",
                "utils.reproducibility"):
        try:
            importlib.import_module(mod)
            print(f"import {mod:<24}✓")
        except Exception as e:                          # noqa: BLE001
            print(f"import {mod:<24}✗  {e}"); ok = False

    if args.forward and ok:
        import tempfile
        from ultralytics import YOLO
        from models.architecture import DASTRPNet
        with tempfile.TemporaryDirectory() as d:
            w = Path(d) / "yolo11n_random.pt"
            YOLO("yolo11n.yaml").save(str(w))      # random weights, no download
            model = DASTRPNet(str(w)).eval()
            x = torch.randn(1, 4, 3, 256, 256)
            boxes = [[torch.tensor([[0.5, 0.5, 0.1, 0.2]])] * 4]
            with torch.no_grad():
                out = model(x, domain_ids=torch.tensor([0]), gt_boxes_seq=boxes)
            print(f"forward pass ✓  risk_prob={out['risk_prob'].numpy().round(3)} "
                  f"uncertainty={out['uncertainty'].item():.3f}")

    print("\nEnvironment OK" if ok else "\nEnvironment has problems (see ✗ above)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
