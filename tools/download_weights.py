"""
tools/download_weights.py
=========================
Download the pre-trained weights listed in weights/manifest.json from the
GitHub Release of this repository and verify their SHA-256 checksums.

Usage
  python tools/download_weights.py                      # all files
  python tools/download_weights.py --only dastrp_exp3.pth yolo11n_motor.pt
  python tools/download_weights.py --base-url <mirror>  # e.g. Zenodo

Only the Python standard library is used.
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = "SaveVic/dastrp-model"
ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dst: Path):
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {dst.name}: {done / 1e6:6.1f} / {total / 1e6:.1f} MB",
                      end="", flush=True)
    print()
    tmp.replace(dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(ROOT / "weights" / "manifest.json"))
    ap.add_argument("--out", default=str(ROOT / "weights"))
    ap.add_argument("--base-url", default=None,
                    help="Default: https://github.com/<repo>/releases/download/<tag>")
    ap.add_argument("--only", nargs="*", default=None, help="Subset of file names")
    ap.add_argument("--force", action="store_true", help="Re-download existing files")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)
    files = manifest.get("files", {})
    if not files:
        sys.exit("manifest.json lists no files yet – weights have not been released.")

    base = (args.base_url or
            f"https://github.com/{REPO}/releases/download/{manifest['release']}").rstrip("/")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    names = args.only or sorted(files)
    failed = []
    for name in names:
        if name not in files:
            print(f"[skip] {name}: not in manifest")
            continue
        dst, expected = out / name, files[name]["sha256"]
        if dst.exists() and not args.force and sha256(dst) == expected:
            print(f"[ok]   {name} (already present)")
            continue
        print(f"[get]  {base}/{name}")
        try:
            download(f"{base}/{name}", dst)
        except Exception as e:                       # noqa: BLE001
            print(f"[fail] {name}: {e}")
            failed.append(name)
            continue
        if sha256(dst) != expected:
            print(f"[fail] {name}: checksum mismatch")
            dst.unlink()
            failed.append(name)
        else:
            print(f"[ok]   {name}")

    if failed:
        sys.exit(f"{len(failed)} file(s) failed: {failed}")
    print(f"All weights in {out}")


if __name__ == "__main__":
    main()
