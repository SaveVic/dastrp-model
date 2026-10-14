"""
tools/check_release.py
======================
Pre-publication check: make sure nothing private goes to the public
repository or to the release assets.

Scans every file that would be committed (`git ls-files` + untracked,
non-ignored files; or a directory walk when there is no git repo) for

  * absolute machine paths (home, media, mount or drive-letter paths)
  * e-mail addresses
  * credentials: private keys, cloud / GitHub / HF / OpenAI / Slack tokens,
    password or secret assignments, .env files
  * private IP addresses
  * files that must not be in git: checkpoints, videos, archives, > 10 MB
  * notebooks saved with outputs
  * images carrying EXIF / GPS / text metadata

With --weights it also scans the release weight files (pickled metadata
only) for absolute paths.

Usage
  python tools/check_release.py
  python tools/check_release.py --weights weights/
Exit code 0 = clean, 1 = findings.
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints",
             "runs", "imbd-esr", "processed", "clips"}
FORBIDDEN_EXT = {".pt", ".pth", ".ckpt", ".onnx", ".engine", ".mp4", ".avi",
                 ".mov", ".zip", ".tar", ".gz", ".7z", ".npz", ".npy", ".h5",
                 ".pkl", ".db", ".sqlite"}
MAX_BYTES = 10 * 1024 * 1024
TEXT_EXT = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".cff", ".cfg",
            ".toml", ".ini", ".sh", ".ipynb", ".tex", ".csv", ""}

PATTERNS = {
    "absolute path": re.compile(
        r"(?<![\w.:/-])/(?:home|media|mnt|Users|root|scratch|srv|data\d*)/[^\s\"'`]+"
        r"|(?<![A-Za-z])[A-Za-z]:\\(?:Users|Documents|Data|Projects)[^\s\"'`]*"),
    "e-mail": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "AWS key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "GitHub token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b|github_pat_[A-Za-z0-9_]{30,}"),
    "HuggingFace token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "OpenAI/Anthropic key": re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}\b"),
    "Slack token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "password/secret": re.compile(
        r"(?i)\b(?:password|passwd|pwd|secret|api_?key|access_?token)\b\s*[:=]\s*['\"][^'\"\s]{4,}['\"]"),
    "private IP": re.compile(
        r"\b(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}\b"),
}
# Allowed matches (documentation examples, public addresses)
ALLOW = [
    re.compile(r"noreply@"),
    re.compile(r"example\.(?:com|org)"),
]


def candidate_files() -> list:
    try:
        out = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout
        files = [ROOT / f for f in out.splitlines() if f]
        if files:
            return [f for f in files if f.is_file()]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    files = []
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if rel.parts[0] == "weights" and p.name not in ("README.md", "manifest.json"):
            continue
        if p.is_file():
            files.append(p)
    return files


def scan_text(path: Path, text: str, findings: list):
    for lineno, line in enumerate(text.splitlines(), 1):
        if "release-check: allow" in line:      # documented, intentional
            continue
        for name, rx in PATTERNS.items():
            for m in rx.finditer(line):
                if any(a.search(m.group(0)) for a in ALLOW):
                    continue
                findings.append((path, lineno, name, m.group(0)[:80]))


def scan_notebook(path: Path, findings: list):
    nb = json.loads(path.read_text(encoding="utf-8"))
    n_out = sum(len(c.get("outputs", [])) for c in nb.get("cells", []))
    if n_out:
        findings.append((path, 0, "notebook has outputs",
                         f"{n_out} output(s) – clear before committing"))
    src = "\n".join("".join(c.get("source", "")) for c in nb.get("cells", []))
    scan_text(path, src, findings)


def scan_image(path: Path, findings: list):
    try:
        from PIL import Image
    except ImportError:
        return
    with Image.open(path) as im:
        meta = {k for k in im.info if k not in ("dpi", "gamma", "transparency",
                                                 "srgb", "aspect", "jfif",
                                                 "jfif_version", "jfif_unit",
                                                 "jfif_density", "progressive",
                                                 "progression")}
        exif = im.getexif() if hasattr(im, "getexif") else {}
        if meta or len(exif):
            findings.append((path, 0, "image metadata",
                             f"keys={sorted(meta)} exif_tags={len(exif)}"))


def scan_weights(wdir: Path, findings: list):
    sys.path.insert(0, str(ROOT / "tools"))
    from export_checkpoint import find_abs_paths
    for p in sorted(wdir.iterdir()):
        if p.suffix in {".pt", ".pth", ".json"}:
            for hit in find_abs_paths(p)[:3]:
                findings.append((p, 0, "absolute path in weights", hit))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None,
                    help="Also scan release weight files in this folder")
    args = ap.parse_args()

    findings = []
    files = candidate_files()
    for p in files:
        rel = p.relative_to(ROOT)
        size = p.stat().st_size
        if p.suffix.lower() in FORBIDDEN_EXT:
            findings.append((rel, 0, "file type not allowed in git", p.suffix))
        if size > MAX_BYTES:
            findings.append((rel, 0, "file too large", f"{size / 1e6:.1f} MB"))
        if p.name == ".env" or p.name.startswith(".env."):
            findings.append((rel, 0, "environment file", p.name))
        if p.suffix == ".ipynb":
            scan_notebook(p, findings)
        elif p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            scan_image(p, findings)
        elif p.suffix.lower() in TEXT_EXT or p.name in ("Makefile", "LICENSE"):
            try:
                scan_text(rel, p.read_text(encoding="utf-8"), findings)
            except UnicodeDecodeError:
                pass

    if args.weights:
        scan_weights(Path(args.weights), findings)

    print(f"Scanned {len(files)} file(s) under {ROOT}")
    if not findings:
        print("✓ No private data, credentials or forbidden files found.")
        return 0
    print(f"✗ {len(findings)} finding(s):")
    for path, line, kind, what in findings:
        pp = Path(path)
        loc = str(pp.relative_to(ROOT) if pp.is_absolute() and pp.is_relative_to(ROOT) else pp)
        loc += f":{line}" if line else ""
        print(f"  [{kind}] {loc}  {what}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
