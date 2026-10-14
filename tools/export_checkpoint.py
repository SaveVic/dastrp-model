"""
tools/export_checkpoint.py
==========================
Turn training checkpoints into clean, public release weights.

What it does
  * DASTRP-Net / baseline checkpoint (best_model.pth):
      - keeps only `model_state`, `epoch`, `val_acc` and the run config
      - drops optimizer / AMP-scaler state (≈2/3 of the file size)
      - rewrites every machine-specific path in the config
        (dataset_root, yolo_weights, save_dir, resume, ...) to a
        repository-relative default, so no local user name / drive
        label / directory layout is published
  * YOLO detector checkpoint (ultralytics best.pt):
      - keeps the (EMA) model only, drops optimizer and git metadata
      - scrubs absolute paths in train_args and in the model object
  * writes weights/manifest.json with file size and SHA-256 for
    tools/download_weights.py, and refuses to finish if any absolute
    path is still found inside an exported file.

Usage
  python tools/export_checkpoint.py \
      --run   runs/exp3 \
      --yolo  runs/yolo/v11n/weights/best.pt \
      --name  dastrp_exp3 \
      --out   weights

  # optional extra runs (ablations / baselines) into the same manifest
  python tools/export_checkpoint.py --run runs/baseline/b2 --name baseline_b2 --out weights

Then upload the files in `weights/` (except manifest.json and README.md)
as assets of a GitHub Release and commit `weights/manifest.json`.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

# Values written into the public config instead of local paths.
PUBLIC_PATHS = {
    "dataset_root": "data/processed/dastrp",
    "yolo_weights": "weights/yolo11n_motor.pt",
    "save_dir":     None,      # filled with runs/<name>
    "resume":       None,
}

ABS_PATH_RE = re.compile(
    # unix absolute paths not preceded by a path/name character
    # (excludes torch-internal names such as "archive/data/0" and URLs)
    rb"(?<![A-Za-z0-9_.:/-])/(?:home|media|mnt|Users|root|tmp|data|scratch|opt|srv|var|run)/"
    # Windows drive paths, e.g. C:\\ or D:/ (but not "https://")
    rb"|(?<![A-Za-z])[A-Za-z]:(?:\\|/(?!/))"
)


def _looks_like_path(s: str) -> bool:
    return (s.startswith(("/", "~")) or re.match(r"^[A-Za-z]:[\\/]", s) is not None
            or ("/" in s and Path(s).suffix in {".pt", ".pth", ".yaml", ".json", ".mp4"}))


def scrub(obj, depth: int = 0):
    """Recursively replace path-like strings by their basename."""
    if depth > 6:
        return obj
    if isinstance(obj, str):
        return Path(obj).name if _looks_like_path(obj) else obj
    if isinstance(obj, dict):
        return {k: scrub(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(scrub(v, depth + 1) for v in obj)
    return obj


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _metadata_bytes(path: Path) -> bytes:
    """Bytes that can carry strings: the pickled metadata of a torch archive
    (tensor storages are skipped – they are raw numbers),
    or the whole file for plain JSON/text files."""
    import zipfile
    if zipfile.is_zipfile(path):
        out = []
        with zipfile.ZipFile(path) as z:
            for n in z.namelist():
                if re.search(r"/data/\d+$", n):  # release-check: allow
                    continue
                out.append(n.encode())
                out.append(z.read(n))
        return b"\n".join(out)
    return path.read_bytes()


def find_abs_paths(path: Path) -> list:
    data = _metadata_bytes(path)
    return [data[m.start(): m.start() + 80].split(b"\x00")[0]
            .decode(errors="replace") for m in ABS_PATH_RE.finditer(data)]


def assert_no_abs_paths(path: Path):
    hits = find_abs_paths(path)
    if hits:
        raise RuntimeError(f"{path}: absolute path(s) still present: {hits[:5]}")


# ── DASTRP-Net / baseline ────────────────────────────────────────

def export_run(run_dir: Path, name: str, out_dir: Path, fp16: bool) -> list:
    ckpt_path = run_dir / "best_model.pth"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = dict(ckpt.get("cfg") or {})
    cfg_file = run_dir / "config.json"
    if cfg_file.exists():
        with open(cfg_file) as f:
            cfg = {**json.load(f), **cfg}

    cfg = scrub(cfg)
    for key, value in PUBLIC_PATHS.items():
        if key in cfg:
            cfg[key] = value
    cfg["save_dir"] = f"runs/{name}"

    state = ckpt["model_state"]
    # strip torch.compile prefix if present
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    if fp16:
        state = {k: (v.half() if torch.is_floating_point(v) else v)
                 for k, v in state.items()}

    public = {
        "model_state": state,
        "cfg":         cfg,
        "epoch":       ckpt.get("epoch"),
        "val_acc":     ckpt.get("val_acc"),
        "format":      "dastrp-release-v1",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    pth = out_dir / f"{name}.pth"
    torch.save(public, pth)
    cfg_out = out_dir / f"{name}_config.json"
    with open(cfg_out, "w") as f:
        json.dump(cfg, f, indent=2)

    files = [pth, cfg_out]
    for extra in ("result_test.json", "result_test_per_video.json"):
        if (run_dir / extra).exists():
            with open(run_dir / extra) as f:
                res = scrub(json.load(f))
            p = out_dir / f"{name}_{extra}"
            with open(p, "w") as f:
                json.dump(res, f, indent=2)
            files.append(p)
    return files


# ── YOLO detector ─────────────────────────────────────────────────

def export_yolo(yolo_path: Path, out_dir: Path, out_name: str) -> list:
    ckpt = torch.load(yolo_path, map_location="cpu", weights_only=False)
    if ckpt.get("ema") is not None:
        ckpt["model"] = ckpt["ema"]
    for k in ("ema", "optimizer", "updates", "git", "train_results",
              "stopper", "scaler", "modelopt"):
        ckpt.pop(k, None)
    if "train_args" in ckpt:
        ckpt["train_args"] = scrub(ckpt["train_args"])

    model = ckpt["model"]
    if hasattr(model, "args"):
        model.args = scrub(model.args) if isinstance(model.args, dict) else model.args
    for attr in ("pt_path", "ckpt_path", "yaml_file"):
        if isinstance(getattr(model, attr, None), str):
            setattr(model, attr, Path(getattr(model, attr)).name)
    if isinstance(getattr(model, "yaml", None), dict):
        model.yaml = scrub(model.yaml)
    model.half()
    for p in model.parameters():
        p.requires_grad = False

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / out_name
    torch.save(ckpt, out)
    return [out]


def update_manifest(out_dir: Path, files: list, tag: str):
    man_path = out_dir / "manifest.json"
    manifest = {"release": tag, "files": {}}
    if man_path.exists():
        with open(man_path) as f:
            manifest = json.load(f)
        manifest["release"] = tag
    for p in files:
        manifest["files"][p.name] = {
            "sha256": sha256(p),
            "bytes":  p.stat().st_size,
        }
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return man_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="Run directory with best_model.pth + config.json")
    ap.add_argument("--name", help="Public name, e.g. dastrp_exp3")
    ap.add_argument("--yolo", help="Ultralytics best.pt of the detector")
    ap.add_argument("--yolo-name", default="yolo11n_motor.pt")
    ap.add_argument("--out", default="weights")
    ap.add_argument("--tag", default="v1.0.0", help="GitHub release tag")
    ap.add_argument("--fp16", action="store_true",
                    help="Store DASTRP-Net weights in half precision")
    args = ap.parse_args()

    if not args.run and not args.yolo:
        ap.error("give --run and/or --yolo")

    out_dir = Path(args.out)
    files = []
    if args.yolo:
        files += export_yolo(Path(args.yolo), out_dir, args.yolo_name)
    if args.run:
        if not args.name:
            ap.error("--name is required with --run")
        files += export_run(Path(args.run), args.name, out_dir, args.fp16)

    for p in files:
        assert_no_abs_paths(p)
    man = update_manifest(out_dir, files, args.tag)

    print("Exported (no absolute paths found):")
    for p in files:
        print(f"  {p}  ({p.stat().st_size / 1e6:.1f} MB)")
    print(f"Manifest: {man}")


if __name__ == "__main__":
    main()
