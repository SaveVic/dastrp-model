"""
prepare_dataset.py
==================
Mempersiapkan folder dataset DASTRP-Net agar siap untuk training dan evaluasi.

Perubahan dari versi sebelumnya:
  - Menambahkan dukungan --split-result untuk membaca split_result.json
    dari split_videos.py, sehingga struktur folder selection_root
    dibangun otomatis dari hasil split
  - Tidak perlu lagi membuat folder annotations/selected secara manual

Alur kerja baru:
  1. Jalankan split_videos.py → menghasilkan split_result.json
  2. Jalankan prepare_dataset.py --split-result split_result.json
     → membuat annotations/selected/{train,val,test}/{video}/frames.json
     → mengekstrak frame dari video
     → menghasilkan dataset/dastrp/{train,val,test}/{images,labels,tracks}
"""

import os
import cv2
import json
import random
import shutil
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


# ─────────────────────────────────────────────────────────────
# Konfigurasi kelas
# ─────────────────────────────────────────────────────────────

CLASS_MAP_YOLO  = {"gerak": 0, "diam": 0}
CLASS_MAP_TRACK = {"gerak": 0, "diam": 1}
RISK_MAP        = {"safe": 0, "warning": 1, "danger": 2}

DATASET_YAML_TEMPLATE_YOLO = """\
# Dataset config untuk YOLOv8
# Di-generate otomatis oleh prepare_dataset.py

path: {root}

train: train/images
val:   val/images
test:  test/images

nc: 1
names:
  0: motor
"""

DATASET_YAML_TEMPLATE_TRACK = """\
# Dataset config untuk DASTRP-Net
# Di-generate otomatis oleh prepare_dataset.py

path: {root}

train: train/images
val:   val/images
test:  test/images

nc: 2
names:
  0: gerak
  1: diam
"""


# ─────────────────────────────────────────────────────────────
# Helper: JSON
# ─────────────────────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────
# NEU: Bangun selection_root dari split_result.json
# ─────────────────────────────────────────────────────────────

def build_selection_from_split_result(
    split_result_path: Path,
    ann_root:          Path,
    selection_root:    Path,
    margin:            int   = 16,
    bg_ratio:          float = 0.1,
    seed:              int   = 42,
    overwrite:         bool  = False,
):
    """
    Baca split_result.json dan bangun struktur:
      selection_root/
        train/{video}/frames.json
        val/{video}/frames.json
        test/{video}/frames.json

    Untuk split train: frame berobjek + margin + bg_ratio frame kosong.
    Untuk split val/test: semua frame.

    Args:
        split_result_path : path ke split_result.json dari split_videos.py
        ann_root          : folder annotations/full
        selection_root    : folder output selection
        margin            : offset frame sebelum/sesudah segmen berobjek (train)
        bg_ratio          : proporsi frame kosong yang diambil (train)
        seed              : random seed
        overwrite         : timpa frames.json yang sudah ada
    """
    if not split_result_path.exists():
        raise FileNotFoundError(
            f"split_result.json tidak ditemukan: {split_result_path}\n"
            f"Jalankan split_videos.py terlebih dahulu."
        )

    result = load_json(split_result_path)

    # Dukung dua format output split_result.json:
    # format baru dari split_videos.py: {"assignment": {video: split}, ...}
    # format lama/manual: {video: split}
    if "assignment" in result:
        assignment = result["assignment"]
    else:
        assignment = result

    print(f"Membaca split dari: {split_result_path}")
    print(f"Total video: {len(assignment)}")

    created = defaultdict(list)
    skipped = []

    for video_name, split in sorted(assignment.items()):
        out_dir    = selection_root / split / video_name
        frames_out = out_dir / "frames.json"

        if frames_out.exists() and not overwrite:
            skipped.append(video_name)
            continue

        # Baca anotasi video
        ann_path = ann_root / f"{video_name}.json"
        if not ann_path.exists():
            print(f"  ⚠ Anotasi tidak ditemukan: {ann_path} — skip")
            continue

        ann_json    = load_json(ann_path)
        start_frame = ann_json.get("start", 0)
        stop_frame  = ann_json.get("stop",  0)

        if split in ("val", "test"):
            # Val/test: semua frame
            frames = list(range(start_frame, stop_frame + 1))
        else:
            # Train: frame berobjek + margin + bg_ratio kosong
            frames = generate_train_frames(
                ann_json,
                margin   = margin,
                bg_ratio = bg_ratio,
                seed     = seed,
            )

        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(frames, frames_out)
        created[split].append(video_name)

        print(f"  ✓ {video_name:<20s} → {split:<6s}  "
              f"({len(frames)} frame)")

    if skipped:
        print(f"\n  [SKIP] {len(skipped)} video sudah ada "
              f"(gunakan overwrite=True untuk menimpa): "
              f"{skipped}")

    print(f"\nStruktur selection_root dibuat:")
    for split in ["train", "val", "test"]:
        vids = created[split]
        if vids:
            print(f"  {split:<6}: {vids}")

    return selection_root


# ─────────────────────────────────────────────────────────────
# Sampling frame untuk training
# ─────────────────────────────────────────────────────────────

def generate_train_frames(
    ann_json:  dict,
    margin:    int   = 16,
    bg_ratio:  float = 0.1,
    seed:      int   = 42,
) -> list:
    """
    Frame berobjek + offset margin kiri-kanan + bg_ratio frame kosong.
    """
    start_frame = ann_json.get("start", 0)
    stop_frame  = ann_json.get("stop",  0)

    # Kumpulkan frame yang punya anotasi
    f_ann = set()
    for entry in ann_json.get("data", []):
        if entry.get("annotations"):
            f_ann.add(int(entry["frame"]))

    # Ekspansi dengan margin
    f_active = set()
    for f in f_ann:
        window_start = max(start_frame, f - margin)
        window_stop  = min(stop_frame,  f + margin)
        f_active.update(range(window_start, window_stop + 1))

    # Sample frame kosong
    all_frames = set(range(start_frame, stop_frame + 1))
    f_bg       = all_frames - f_active

    random.seed(seed)
    f_bg_list    = list(f_bg)
    num_bg       = int(len(f_bg_list) * bg_ratio)
    f_bg_sampled = set(random.sample(f_bg_list, num_bg))

    return sorted(list(f_active | f_bg_sampled))


# ─────────────────────────────────────────────────────────────
# Konversi anotasi
# ─────────────────────────────────────────────────────────────

def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def annotation_to_yolo(ann: dict, is_yolo: bool = False) -> str | None:
    label     = ann.get("label", "").strip().lower()
    CLASS_MAP = CLASS_MAP_YOLO if is_yolo else CLASS_MAP_TRACK
    if label not in CLASS_MAP:
        return None
    class_id = CLASS_MAP[label]
    cx = _clamp(float(ann["x"]))
    cy = _clamp(float(ann["y"]))
    w  = _clamp(float(ann["w"]))
    h  = _clamp(float(ann["h"]))
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def annotation_to_track(ann: dict, idx: int) -> dict | None:
    label = ann.get("label", "").strip().lower()
    if label not in CLASS_MAP_TRACK:
        return None
    risk_raw = ann.get("risk", "safe")
    risk_str = risk_raw.strip().lower() if isinstance(risk_raw, str) else "safe"
    risk_str = {
        "warning": "warning", "warn": "warning",
        "danger":  "danger",  "bahaya": "danger",
        "safe":    "safe",    "aman":   "safe",
    }.get(risk_str, "safe")
    return {
        "idx":      idx,
        "id":       ann.get("id"),
        "label":    label,
        "class_id": CLASS_MAP_TRACK[label],
        "risk":     risk_str,
        "risk_id":  RISK_MAP.get(risk_str, 0),
        "cx":       round(_clamp(float(ann["x"])), 6),
        "cy":       round(_clamp(float(ann["y"])), 6),
        "w":        round(_clamp(float(ann["w"])), 6),
        "h":        round(_clamp(float(ann["h"])), 6),
    }


def build_ann_lookup(ann_json: dict) -> dict:
    lookup = {}
    for entry in ann_json.get("data", []):
        frame_idx = int(entry["frame"])
        anns      = entry.get("annotations", [])
        if anns:
            lookup[frame_idx] = anns
    return lookup


# ─────────────────────────────────────────────────────────────
# Core: proses satu video
# ─────────────────────────────────────────────────────────────

def process_video(
    video_name:    str,
    split:         str,
    frames_json:   Path,
    ann_json_path: Path,
    video_path:    Path,
    out_img_dir:   Path,
    out_lbl_dir:   Path,
    out_trk_dir:   Path,
    jpg_quality:   int  = 95,
    is_yolo:       bool = False,
) -> dict:

    if not ann_json_path.exists():
        raise FileNotFoundError(f"Anotasi tidak ditemukan: {ann_json_path}")

    ann_json   = load_json(ann_json_path)
    ann_lookup = build_ann_lookup(ann_json)

    # ── Frame list ────────────────────────────────────────────
    if frames_json.exists():
        frames_to_use = load_json(frames_json)
    else:
        # Fallback: generate otomatis jika frames.json belum ada
        start_frame = ann_json.get("start", 0)
        stop_frame  = ann_json.get("stop",  0)
        if split in ("val", "test"):
            frames_to_use = list(range(start_frame, stop_frame + 1))
        else:
            frames_to_use = generate_train_frames(ann_json, margin=16, bg_ratio=0.1)
        save_json(frames_to_use, frames_json)

    if not frames_to_use:
        raise ValueError(f"Tidak ada frame untuk {video_name}")

    frames_set = set(frames_to_use)
    min_frame  = min(frames_set)
    max_frame  = max(frames_set)

    # ── Buka video ────────────────────────────────────────────
    if not video_path.exists():
        raise FileNotFoundError(f"Video tidak ditemukan: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)
    if not is_yolo:
        out_trk_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "video":            video_name,
        "split":            split,
        "total_frames":     total_frames,
        "fps":              fps,
        "resolution":       f"{width}x{height}",
        "frames_selected":  len(frames_to_use),
        "frames_with_obj":  0,
        "frames_empty":     0,
        "objects_gerak":    0,
        "objects_diam":     0,
        "objects_motor":    0,
        "risk_safe":        0,
        "risk_warning":     0,
        "risk_danger":      0,
        "unique_track_ids": set(),
        "skipped":          0,
    }

    # ── Ekstraksi frame sekuensial ────────────────────────────
    if min_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min_frame)

    curr_idx = min_frame
    pbar = tqdm(total=len(frames_set),
                desc=f"  {video_name} [{split}]", leave=False)

    while curr_idx <= max_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if curr_idx in frames_set:
            stem     = f"{video_name}__{curr_idx:06d}"
            img_path = out_img_dir / f"{stem}.jpg"
            lbl_path = out_lbl_dir / f"{stem}.txt"
            trk_path = out_trk_dir / f"{stem}.json"

            cv2.imwrite(str(img_path), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])

            anns          = ann_lookup.get(curr_idx, [])
            yolo_lines    = []
            track_objects = []

            for idx, ann in enumerate(anns):
                yolo_line = annotation_to_yolo(ann, is_yolo)
                if not yolo_line:
                    continue
                yolo_lines.append(yolo_line)

                if not is_yolo:
                    track_obj = annotation_to_track(ann, idx)
                    if not track_obj:
                        continue
                    track_objects.append(track_obj)
                    label = track_obj["label"]
                    if label == "gerak":
                        stats["objects_gerak"] += 1
                    elif label == "diam":
                        stats["objects_diam"]  += 1
                    stats[f"risk_{track_obj['risk']}"] += 1
                    obj_id = track_obj["id"]
                    if obj_id is not None:
                        stats["unique_track_ids"].add(obj_id)
                else:
                    stats["objects_motor"] += 1

            with open(lbl_path, "w") as f:
                f.write("\n".join(yolo_lines))
                if yolo_lines:
                    f.write("\n")

            if not is_yolo:
                with open(trk_path, "w") as f:
                    json.dump({
                        "video":   video_name,
                        "frame":   curr_idx,
                        "objects": track_objects,
                    }, f)

            if yolo_lines:
                stats["frames_with_obj"] += 1
            else:
                stats["frames_empty"] += 1
            pbar.update(1)

        curr_idx += 1

    pbar.close()
    cap.release()
    stats["unique_track_ids"] = len(stats["unique_track_ids"])
    return stats


# ─────────────────────────────────────────────────────────────
# Scan split
# ─────────────────────────────────────────────────────────────

def scan_split(split_dir: Path) -> list:
    ignore_dirs = {"images", "labels", "tracks"}
    return [d.name for d in sorted(split_dir.iterdir())
            if d.is_dir() and d.name not in ignore_dirs]


# ─────────────────────────────────────────────────────────────
# Main routine
# ─────────────────────────────────────────────────────────────

def prepare_dataset(
    dataset_root:      str,
    ann_root:          str,
    selection_root:    str,
    video_root:        str,
    split_result_path: str  = None,
    jpg_quality:       int  = 95,
    overwrite:         bool = False,
    is_yolo:           bool = False,
    margin:            int  = 16,
    bg_ratio:          float = 0.1,
    seed:              int  = 42,
):
    dataset_root      = Path(dataset_root)
    ann_root          = Path(ann_root)
    selection_root    = Path(selection_root)
    video_root        = Path(video_root)

    print("=" * 60)
    print("DATASET PREPARATION —",
          "YOLO" if is_yolo else "DASTRP-Net")
    print("=" * 60)
    print(f"Dataset root : {dataset_root.resolve()}")
    print(f"Annotations  : {ann_root.resolve()}")
    print(f"Selection    : {selection_root.resolve()}")
    print(f"Videos       : {video_root.resolve()}")
    if split_result_path:
        print(f"Split result : {split_result_path}")
    print()

    # ── Step 1: Bangun selection_root dari split_result.json ──
    if split_result_path:
        print("Membangun selection dari split_result.json...")
        build_selection_from_split_result(
            split_result_path = Path(split_result_path),
            ann_root          = ann_root,
            selection_root    = selection_root,
            margin            = margin,
            bg_ratio          = bg_ratio,
            seed              = seed,
            overwrite         = overwrite,
        )
        print()

    # ── Step 2: Ekstraksi frame ───────────────────────────────
    splits     = ["train", "val", "test"]
    all_stats  = []
    global_meta = defaultdict(lambda: {
        "videos": 0, "frames_selected": 0, "frames_with_obj": 0,
        "frames_empty": 0, "objects_gerak": 0, "objects_diam": 0,
        "objects_motor": 0,
    })

    for split in splits:
        split_sel_dir    = selection_root / split
        target_split_dir = dataset_root   / split

        if not split_sel_dir.exists():
            print(f"[SKIP] Split '{split}' tidak ada di {split_sel_dir}")
            continue

        out_img = target_split_dir / "images"
        out_lbl = target_split_dir / "labels"
        out_trk = target_split_dir / "tracks"

        if not overwrite and out_img.exists() and any(out_img.iterdir()):
            n_exist = len(list(out_img.glob("*.jpg")))
            print(f"[SKIP] {split}/images sudah ada ({n_exist} file). "
                  f"Gunakan --overwrite untuk menimpa.")
            continue

        if overwrite:
            for d in [out_img, out_lbl, out_trk]:
                if d.exists():
                    shutil.rmtree(d)

        videos = scan_split(split_sel_dir)
        print(f"[{split.upper()}] {len(videos)} video: {', '.join(videos)}")

        split_stats = []
        for video_name in videos:
            frames_json  = split_sel_dir / video_name / "frames.json"
            ann_path     = ann_root      / f"{video_name}.json"
            video_path   = video_root    / f"{video_name}.mp4"

            try:
                s = process_video(
                    video_name    = video_name,
                    split         = split,
                    frames_json   = frames_json,
                    ann_json_path = ann_path,
                    video_path    = video_path,
                    out_img_dir   = out_img,
                    out_lbl_dir   = out_lbl,
                    out_trk_dir   = out_trk,
                    jpg_quality   = jpg_quality,
                    is_yolo       = is_yolo,
                )
                split_stats.append(s)
                all_stats.append(s)

                g = global_meta[split]
                g["videos"]          += 1
                g["frames_selected"] += s["frames_selected"]
                g["frames_with_obj"] += s["frames_with_obj"]
                g["frames_empty"]    += s["frames_empty"]
                g["objects_gerak"]   += s["objects_gerak"]
                g["objects_diam"]    += s["objects_diam"]
                g["objects_motor"]   += s["objects_motor"]

                if not is_yolo:
                    print(f"  ✓ {video_name}: "
                          f"{s['frames_selected']} frame "
                          f"({s['frames_with_obj']} obj / "
                          f"{s['frames_empty']} kosong) | "
                          f"gerak={s['objects_gerak']} "
                          f"diam={s['objects_diam']}")
                else:
                    print(f"  ✓ {video_name}: "
                          f"{s['frames_selected']} frame "
                          f"({s['frames_with_obj']} obj / "
                          f"{s['frames_empty']} kosong) | "
                          f"motor={s['objects_motor']}")

            except Exception as e:
                print(f"  ✗ {video_name}: {e}")

        save_json(split_stats, target_split_dir / "meta.json")
        print(f"  → meta.json → {target_split_dir / 'meta.json'}\n")

    # ── Step 3: class_names.txt dan dataset.yaml ──────────────
    class_file = dataset_root / "class_names.txt"
    with open(class_file, "w") as f:
        if is_yolo:
            f.write("motor\n")
        else:
            for name in sorted(CLASS_MAP_TRACK, key=CLASS_MAP_TRACK.get):
                f.write(name + "\n")

    yaml_path = dataset_root / "dataset.yaml"
    with open(yaml_path, "w") as f:
        template = (DATASET_YAML_TEMPLATE_YOLO if is_yolo
                    else DATASET_YAML_TEMPLATE_TRACK)
        f.write(template.format(root=str(dataset_root.resolve())))

    # ── Ringkasan ─────────────────────────────────────────────
    print("=" * 60)
    print("RINGKASAN AKHIR")
    print("=" * 60)
    if is_yolo:
        hdr = (f"{'Split':<8} {'Video':>6} {'Frame':>8} "
               f"{'w/Obj':>7} {'Kosong':>8} {'Motor':>7} {'%Kosong':>8}")
    else:
        hdr = (f"{'Split':<8} {'Video':>6} {'Frame':>8} "
               f"{'w/Obj':>7} {'Kosong':>8} "
               f"{'Gerak':>7} {'Diam':>6} {'%Kosong':>8}")
    print(hdr)
    print("-" * len(hdr))

    for split in splits:
        g = global_meta[split]
        if g["videos"] == 0:
            continue
        pct = g["frames_empty"] / max(g["frames_selected"], 1) * 100
        if is_yolo:
            print(f"{split:<8} {g['videos']:>6} {g['frames_selected']:>8} "
                  f"{g['frames_with_obj']:>7} {g['frames_empty']:>8} "
                  f"{g['objects_motor']:>7} {pct:>7.1f}%")
        else:
            print(f"{split:<8} {g['videos']:>6} {g['frames_selected']:>8} "
                  f"{g['frames_with_obj']:>7} {g['frames_empty']:>8} "
                  f"{g['objects_gerak']:>7} {g['objects_diam']:>6} "
                  f"{pct:>7.1f}%")

    print("-" * len(hdr))
    print(f"{'TOTAL':<8} "
          f"{sum(global_meta[s]['videos'] for s in splits):>6} "
          f"{sum(global_meta[s]['frames_selected'] for s in splits):>8} "
          f"{sum(global_meta[s]['frames_with_obj'] for s in splits):>7}\n")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Build the processed DASTRP-Net / YOLO dataset from raw "
                    "CVAT-exported annotations and full-length videos "
                    "(internal pipeline used for the paper). For the public "
                    "IMBD-ESR release use tools/convert_imbd_esr.py instead.")
    ap.add_argument("--dataset-root", required=True,
                    help="Output root, e.g. dataset/dastrp")
    ap.add_argument("--ann-root", required=True,
                    help="Folder with raw per-video annotation JSON (annotations/full)")
    ap.add_argument("--selection-root", required=True,
                    help="Folder for per-split frame selections (annotations/selected)")
    ap.add_argument("--video-root", required=True,
                    help="Folder with the source .mp4 videos")
    ap.add_argument("--split-result", default=None,
                    help="split_result.json produced by tools/split_videos.py")
    ap.add_argument("--quality", type=int, default=100, help="JPEG quality")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--yolo", action="store_true",
                    help="Single-class 'motor' labels for detector training")
    ap.add_argument("--margin", type=int, default=16)
    ap.add_argument("--bg-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    prepare_dataset(
        dataset_root      = a.dataset_root,
        ann_root          = a.ann_root,
        selection_root    = a.selection_root,
        video_root        = a.video_root,
        split_result_path = a.split_result,
        jpg_quality       = a.quality,
        overwrite         = a.overwrite,
        is_yolo           = a.yolo,
        margin            = a.margin,
        bg_ratio          = a.bg_ratio,
        seed              = a.seed,
    )
