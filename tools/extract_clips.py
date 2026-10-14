"""
tools/extract_clips.py
======================
Extract the T-frame clips used by DASTRP-Net from a processed dataset.

It reproduces exactly the clip construction of data/dataset.py
(ClipSampler; train: stride s, val/test: non-overlapping stride T) and
writes, per split:

  <out>/clips_<split>.json   one record per clip:
        clip_id, video, domain, stems, frame_idx,
        frame_risks, clip_label (= max frame risk), n_objects
  <out>/clips_summary.json   clip counts per split / label / domain

Optionally each clip is rendered for visual inspection
(--export jpg: one horizontal strip per clip, --export mp4: short video),
with ground-truth boxes coloured by per-object risk.

Usage
  python tools/extract_clips.py --dataset data/processed/dastrp --out data/clips
  python tools/extract_clips.py --dataset data/processed/dastrp --out data/clips \
      --splits test --export jpg --label danger --limit 50
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.dataset import (ClipSampler, DOMAIN_NAMES, RISK_NAMES,  # noqa: E402
                          aggregate_frame_risk, infer_domain, parse_stem)

RISK_COLOR = {0: (80, 200, 80), 1: (0, 190, 255), 2: (60, 60, 230)}   # BGR
LABEL_ID = {"safe": 0, "warning": 1, "danger": 2}


def load_split(split_dir: Path) -> dict:
    """{video: [(stem, objects), ...]} in temporal order (same rule as loader)."""
    img_dir, trk_dir = split_dir / "images", split_dir / "tracks"
    videos = defaultdict(list)
    for img in sorted(img_dir.glob("*.jpg")):
        trk = trk_dir / f"{img.stem}.json"
        if not trk.exists():
            continue
        with open(trk) as f:
            objects = json.load(f).get("objects", [])
        video, _ = parse_stem(img.stem)
        videos[video].append((img.stem, objects))
    return videos


def build_clips(videos: dict, T: int, stride: int) -> list:
    clips = []
    for video, frames in sorted(videos.items()):
        stems = [s for s, _ in frames]
        objs = dict(frames)
        for clip_stems in ClipSampler(stems, T=T, stride=stride).clips:
            risks = [aggregate_frame_risk(objs[s]) for s in clip_stems]
            clips.append({
                "clip_id":     f"{video}@{parse_stem(clip_stems[0])[1]:06d}",
                "video":       video,
                "domain":      DOMAIN_NAMES.get(infer_domain(video), "unknown"),
                "stems":       clip_stems,
                "frame_idx":   [parse_stem(s)[1] for s in clip_stems],
                "frame_risks": risks,
                "clip_label":  max(risks),
                "n_objects":   [len(objs[s]) for s in clip_stems],
            })
    return clips


def draw(img: np.ndarray, objects: list) -> np.ndarray:
    h, w = img.shape[:2]
    for o in objects:
        cx, cy, bw, bh = (float(o[k]) for k in ("cx", "cy", "w", "h"))
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        r = int(o.get("risk_id", LABEL_ID.get(str(o.get("risk", "safe")), 0)))
        cv2.rectangle(img, (x1, y1), (x2, y2), RISK_COLOR[r], 2)
    return img


def render(clip: dict, split_dir: Path, out_dir: Path, mode: str, height: int):
    frames = []
    for stem in clip["stems"]:
        img = cv2.imread(str(split_dir / "images" / f"{stem}.jpg"))
        with open(split_dir / "tracks" / f"{stem}.json") as f:
            img = draw(img, json.load(f).get("objects", []))
        scale = height / img.shape[0]
        frames.append(cv2.resize(img, (int(img.shape[1] * scale), height)))

    label = RISK_NAMES[clip["clip_label"]]
    name = f"{clip['clip_id'].replace('@', '_')}_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "jpg":
        strip = np.concatenate(frames, axis=1)
        cv2.putText(strip, f"{clip['clip_id']}  clip label: {label}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, RISK_COLOR[clip["clip_label"]], 2)
        cv2.imwrite(str(out_dir / f"{name}.jpg"), strip)
    else:
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(str(out_dir / f"{name}.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 4, (w, h))
        for fr in frames:
            vw.write(fr)
        vw.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="Processed dataset root")
    ap.add_argument("--out", default="data/clips")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--T", type=int, default=4, help="Clip length (frames)")
    ap.add_argument("--stride", type=int, default=2,
                    help="Train stride; val/test always use stride = T")
    ap.add_argument("--export", choices=["none", "jpg", "mp4"], default="none")
    ap.add_argument("--label", choices=["safe", "warning", "danger"], default=None,
                    help="Only render clips with this clip label")
    ap.add_argument("--limit", type=int, default=100,
                    help="Max clips rendered per split (random sample)")
    ap.add_argument("--height", type=int, default=240, help="Render height (px)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root, out = Path(args.dataset), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    summary = {"T": args.T, "train_stride": args.stride, "splits": {}}

    for split in args.splits:
        split_dir = root / split
        if not (split_dir / "images").exists():
            print(f"[skip] {split}: {split_dir} not found")
            continue
        stride = args.stride if split == "train" else args.T
        clips = build_clips(load_split(split_dir), args.T, stride)

        with open(out / f"clips_{split}.json", "w") as f:
            json.dump(clips, f)
        summary["splits"][split] = {
            "clips":    len(clips),
            "videos":   len({c["video"] for c in clips}),
            "stride":   stride,
            "label":    {RISK_NAMES[k]: v for k, v in
                         sorted(Counter(c["clip_label"] for c in clips).items())},
            "domain":   dict(sorted(Counter(c["domain"] for c in clips).items())),
        }
        print(f"[{split}] {len(clips)} clips -> {out / f'clips_{split}.json'}  "
              f"{summary['splits'][split]['label']}")

        if args.export != "none":
            sel = [c for c in clips if args.label is None
                   or c["clip_label"] == LABEL_ID[args.label]]
            sel = rng.sample(sel, min(args.limit, len(sel)))
            for c in sel:
                render(c, split_dir, out / split, args.export, args.height)
            print(f"        rendered {len(sel)} clip(s) as {args.export} -> {out / split}")

    with open(out / "clips_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
