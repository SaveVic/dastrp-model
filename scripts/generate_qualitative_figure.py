"""
scripts/generate_qualitative_figure.py
========================================
Generate figure kualitatif untuk Section V-F (Qualitative Analysis)
IEEE T-ITS paper: grid 3 domain x 2 kondisi (correct/incorrect),
menampilkan clip-level prediction, bounding box per-objek risk,
dan uncertainty score.

Alur kerja:
  1. Jalankan model pada test set, kumpulkan semua clip prediction
  2. Untuk setiap domain (UGM, Utara, Ringroad), cari:
       - satu clip CORRECT dengan confidence tinggi (uncertainty rendah)
       - satu clip INCORRECT (prediksi salah, idealnya Warning/Danger
         yang missed atau Ringroad yang occluded)
  3. Render frame terakhir tiap clip terpilih dengan:
       - bounding box berwarna per-objek risk (hijau/kuning/merah)
       - label clip-level prediction + ground truth
       - uncertainty score
  4. Susun grid 3x2 dan simpan sebagai PNG/PDF resolusi tinggi

Cara pakai:
  python scripts/generate_qualitative_figure.py \
    --config    ".../exp3/config.json" \
    --out-dir   ".../exp3/qualitative" \
    --grid-out  ".../exp3/qualitative/fig_qualitative.png"
"""

import json
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.reproducibility import set_seed, add_seed_args
from models.architecture import DASTRPNet
from data.dataset import get_dataloaders, DOMAIN_NAMES, RISK_NAMES


# Warna BGR (OpenCV) per risk level — konsisten dengan inference.py
RISK_COLOR_BGR = {
    0: (80,  200, 50),    # safe    — hijau
    1: (30,  180, 255),   # warning — kuning
    2: (60,   60, 255),   # danger  — merah
}
RISK_LABEL = {0: "Safe", 1: "Warning", 2: "Danger"}
DOMAIN_ORDER = ["UGM", "Utara", "Ringroad"]


# ─────────────────────────────────────────────
# 1. Load model
# ─────────────────────────────────────────────

def load_model(config_path: Path, device: torch.device) -> tuple:
    with open(config_path) as f:
        cfg = json.load(f)

    save_dir  = Path(cfg["save_dir"])
    ckpt_path = save_dir / "best_model.pth"
    ckpt      = torch.load(ckpt_path, map_location=device)

    model = DASTRPNet(
        yolo_weights = cfg["yolo_weights"],
        yolo_conf    = cfg.get("yolo_conf",   0.25),
        yolo_iou     = cfg.get("yolo_iou",    0.45),
        dist_thresh  = cfg.get("dist_thresh", 0.3),
        T_agg        = cfg.get("T_agg",       None),
        ablation     = cfg.get("ablation",    None),
    ).to(device)

    state_dict = ckpt["model_state"]
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        new_sd = {k.replace("_orig_mod.", ""): v
                  for k, v in state_dict.items()}
        model.load_state_dict(new_sd)
    model.eval()

    return model, cfg


# ─────────────────────────────────────────────
# 2. Cari kandidat sampel per domain
# ─────────────────────────────────────────────

@torch.no_grad()
def find_candidate_samples(
    model:   DASTRPNet,
    loader,
    device:  torch.device,
    use_amp: bool = True,
) -> dict:
    """
    Jalankan model di seluruh test set, kumpulkan kandidat
    correct (uncertainty rendah) dan incorrect per domain.

    Returns dict:
      {
        "UGM":      {"correct": [...], "incorrect": [...]},
        "Utara":    {"correct": [...], "incorrect": [...]},
        "Ringroad": {"correct": [...], "incorrect": [...]},
      }
    Setiap entry list berisi dict candidate:
      {video_name, frame_idx_in_clip, clip_images, pred, label,
       uncertainty, boxes, obj_risk_pred}
    """
    candidates = defaultdict(lambda: {"correct": [], "incorrect": []})

    for batch in loader:
        clip_images = batch["images"].to(device, non_blocking=True)
        domain_ids  = batch["domain_ids"].to(device, non_blocking=True)
        T           = batch["frame_risks"].size(1)
        B           = clip_images.size(0)

        with torch.amp.autocast("cuda",
                                enabled=(use_amp and device.type == "cuda")):
            outputs = model(
                clip_images  = clip_images,
                domain_ids   = None,    # inference mode
                gt_boxes_seq = None,
            )

        preds  = outputs["risk_pred"].cpu()
        labels = batch["frame_risks"].max(dim=1).values
        uncs   = outputs["uncertainty"].cpu()

        for b in range(B):
            d_id   = domain_ids[b].item()
            d_name = DOMAIN_NAMES.get(d_id, "Unknown")
            is_correct = (preds[b].item() == labels[b].item())

            entry = {
                "video_name":    batch["video_names"][b],
                "clip_image_t":  clip_images[b, -1].cpu(),  # frame terakhir (3,H,W)
                "pred":          preds[b].item(),
                "label":         labels[b].item(),
                "uncertainty":   uncs[b].item(),
                "boxes":         outputs["det_boxes"][b].cpu().numpy(),
                "obj_risk_pred": (outputs["obj_risk_pred"][b].cpu().numpy()
                                  if outputs["obj_risk_pred"][b].size(0) > 0
                                  else np.array([])),
            }

            key = "correct" if is_correct else "incorrect"
            candidates[d_name][key].append(entry)

    return candidates


# ─────────────────────────────────────────────
# 3. Seleksi sampel terbaik per kriteria
# ─────────────────────────────────────────────

def select_best_samples(candidates: dict) -> dict:
    """
    Dari kandidat per domain, pilih satu correct dan satu incorrect
    paling representatif:
      - correct  : uncertainty PALING RENDAH di antara prediksi Warning/Danger
                   yang benar (paling meyakinkan), fallback ke Safe jika
                   tidak ada Warning/Danger yang benar
      - incorrect: uncertainty PALING TINGGI (paling "bingung") di antara
                   yang salah, idealnya kelas non-Safe yang missed
    """
    selected = {}

    for d_name in DOMAIN_ORDER:
        d_candidates = candidates.get(d_name, {"correct": [], "incorrect": []})

        # ── Correct sample ──────────────────────────────────────
        correct_list = d_candidates["correct"]
        # Prioritaskan correct non-Safe (lebih menarik secara visual)
        non_safe_correct = [c for c in correct_list if c["label"] != 0]
        pool = non_safe_correct if non_safe_correct else correct_list

        best_correct = (min(pool, key=lambda c: c["uncertainty"])
                        if pool else None)

        # ── Incorrect sample ─────────────────────────────────────
        incorrect_list = d_candidates["incorrect"]
        # Prioritaskan kesalahan pada kelas non-Safe (lebih informatif)
        non_safe_incorrect = [c for c in incorrect_list if c["label"] != 0]
        pool_inc = non_safe_incorrect if non_safe_incorrect else incorrect_list

        best_incorrect = (max(pool_inc, key=lambda c: c["uncertainty"])
                          if pool_inc else None)

        selected[d_name] = {
            "correct":   best_correct,
            "incorrect": best_incorrect,
        }

        print(f"  [{d_name}] "
              f"correct={'found' if best_correct else 'NOT FOUND'} "
              f"(label={RISK_LABEL.get(best_correct['label'],'?') if best_correct else '-'})  "
              f"incorrect={'found' if best_incorrect else 'NOT FOUND'} "
              f"(true={RISK_LABEL.get(best_incorrect['label'],'?') if best_incorrect else '-'}, "
              f"pred={RISK_LABEL.get(best_incorrect['pred'],'?') if best_incorrect else '-'})")

    return selected


# ─────────────────────────────────────────────
# 4. Render satu panel (denormalize + annotate)
# ─────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def tensor_to_bgr(img_t: torch.Tensor) -> np.ndarray:
    """
    (3, H, W) normalized tensor → (H, W, 3) BGR uint8 untuk OpenCV.
    """
    img = img_t.permute(1, 2, 0).numpy()           # (H, W, 3) RGB
    img = img * IMAGENET_STD + IMAGENET_MEAN        # denormalize
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def render_panel(
    entry:      dict,
    panel_size: int = 480,
) -> np.ndarray:
    """
    Render satu panel kualitatif: frame + bbox per-objek + label overlay.

    Returns: (panel_size, panel_size, 3) BGR image siap di-tile.
    """
    img = tensor_to_bgr(entry["clip_image_t"])
    img = cv2.resize(img, (panel_size, panel_size))
    H, W = img.shape[:2]

    boxes     = entry["boxes"]
    obj_risks = entry["obj_risk_pred"]

    # ── Bounding boxes per-objek risk ──────────────────────────
    for i, box in enumerate(boxes):
        cx, cy, bw, bh = box
        x1 = max(0, int((cx - bw/2) * W))
        y1 = max(0, int((cy - bh/2) * H))
        x2 = min(W, int((cx + bw/2) * W))
        y2 = min(H, int((cy + bh/2) * H))

        obj_r = int(obj_risks[i]) if i < len(obj_risks) else entry["pred"]
        color = RISK_COLOR_BGR.get(obj_r, (200, 200, 200))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # ── Label overlay (clip-level) ─────────────────────────────
    pred_name  = RISK_LABEL[entry["pred"]]
    true_name  = RISK_LABEL[entry["label"]]
    is_correct = (entry["pred"] == entry["label"])
    unc        = entry["uncertainty"]

    overlay_color = RISK_COLOR_BGR[entry["pred"]]
    status_color  = (80, 200, 50) if is_correct else (60, 60, 255)
    status_text   = "CORRECT" if is_correct else "INCORRECT"

    # Background bar atas untuk teks
    bar_h = 56
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (W, bar_h), (20, 20, 20), -1)
    img = cv2.addWeighted(overlay, 0.75, img, 0.25, 0)

    cv2.putText(img, f"Pred: {pred_name}  |  GT: {true_name}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                overlay_color, 2, cv2.LINE_AA)
    cv2.putText(img, f"u={unc:.3f}  [{status_text}]",
                (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                status_color, 1, cv2.LINE_AA)

    # Border luar sesuai status correct/incorrect
    cv2.rectangle(img, (0, 0), (W-1, H-1), status_color, 3)

    return img


# ─────────────────────────────────────────────
# 5. Susun grid 3x2 dan simpan
# ─────────────────────────────────────────────

def compose_grid(
    selected:    dict,
    out_dir:     Path,
    grid_path:   Path,
    panel_size:  int = 360,
    label_h:     int = 32,
):
    """
    Susun grid LANDSCAPE: baris=[correct, incorrect], kolom=domain.

    Orientasi ini menghasilkan rasio lebar:tinggi ~3:2, jauh lebih
    sesuai untuk figure* (lebar dua kolom) di paper IEEE dibanding
    orientasi domain-as-rows yang menghasilkan grid portrait.

    Panel default diperkecil ke 360px (dari 480px) karena sekarang
    ada 3 kolom alih-alih 2, total lebar tetap terkendali.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Header kolom (nama domain) — dibuat dulu sebelum baris
    col_header = np.full((label_h, panel_size * len(DOMAIN_ORDER), 3),
                         30, dtype=np.uint8)
    for i, d_name in enumerate(DOMAIN_ORDER):
        x = i * panel_size + panel_size // 2 - 30
        cv2.putText(col_header, d_name, (x, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    rows = []
    row_labels = ["correct", "incorrect"]
    row_label_text = {"correct": "Correct Prediction",
                       "incorrect": "Incorrect Prediction"}

    for key in row_labels:
        panels = []
        for d_name in DOMAIN_ORDER:
            entry = selected[d_name][key]
            if entry is None:
                panel = np.full((panel_size, panel_size, 3), 40, dtype=np.uint8)
                cv2.putText(panel, "N/A",
                            (panel_size//2-30, panel_size//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150,150,150), 2)
            else:
                panel = render_panel(entry, panel_size)
                fname = f"{d_name.lower()}_{key}.png"
                cv2.imwrite(str(out_dir / fname), panel)
                print(f"  Panel disimpan: {out_dir / fname}")
            panels.append(panel)

        row_img = np.hstack(panels)

        # Label baris (Correct/Incorrect) di kiri
        row_label_w = 140
        label_strip = np.full((panel_size, row_label_w, 3), 30, dtype=np.uint8)
        text_color = (80, 200, 50) if key == "correct" else (60, 60, 255)
        # Teks vertikal disimulasikan dengan rotasi
        label_canvas = np.full((row_label_w, panel_size, 3), 30, dtype=np.uint8)
        cv2.putText(label_canvas, row_label_text[key], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2)
        label_rot = cv2.rotate(label_canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)

        rows.append(np.hstack([label_rot, row_img]))

    # Gabungkan header dengan offset label kiri
    header_padded = np.hstack([
        np.full((label_h, row_label_w, 3), 30, dtype=np.uint8),
        col_header,
    ])

    grid = np.vstack([header_padded] + rows)

    grid_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(grid_path), grid)
    print(f"\nGrid kualitatif tersimpan → {grid_path}")
    print(f"  Resolusi: {grid.shape[1]}×{grid.shape[0]}  "
          f"(rasio {grid.shape[1]/grid.shape[0]:.2f}:1)")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate figure kualitatif untuk paper"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path ke config.json hasil training (mis. exp3)"
    )
    parser.add_argument(
        "--out-dir", required=True,
        help="Folder untuk simpan panel individual"
    )
    parser.add_argument(
        "--grid-out", default=None,
        help="Path output grid gabungan "
             "(default: {out_dir}/fig_qualitative.png)"
    )
    parser.add_argument(
        "--panel-size", type=int, default=360,
        help="Ukuran tiap panel persegi (px) — default 360 untuk layout 3-kolom"
    )
    parser.add_argument(
        "--split", default="test", choices=["val", "test"],
        help="Split yang digunakan untuk mencari sampel"
    )
    add_seed_args(parser)
    args = parser.parse_args()
    set_seed(args.seed, args.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model, cfg = load_model(Path(args.config), device)

    print(f"\nMemuat dataset split='{args.split}'...")
    loaders = get_dataloaders(
        dataset_root = cfg["dataset_root"],
        T            = cfg.get("T", 4),
        stride       = cfg.get("T", 4),
        img_size     = cfg.get("img_size", 640),
        batch_size   = cfg.get("batch_size", 16),
        num_workers  = cfg.get("num_workers", 4),
        seed         = cfg.get("seed", 42),
        splits       = [args.split],
    )

    print(f"\nMencari kandidat sampel per domain...")
    candidates = find_candidate_samples(
        model, loaders[args.split], device,
        use_amp=cfg.get("use_amp", True),
    )

    print(f"\nMenyeleksi sampel terbaik...")
    selected = select_best_samples(candidates)

    out_dir   = Path(args.out_dir)
    grid_path = (Path(args.grid_out) if args.grid_out
                 else out_dir / "fig_qualitative.png")

    print(f"\nMenyusun grid...")
    compose_grid(selected, out_dir, grid_path, args.panel_size)

    # Simpan metadata seleksi untuk referensi caption
    meta = {}
    for d_name, sel in selected.items():
        meta[d_name] = {}
        for key, entry in sel.items():
            if entry is not None:
                meta[d_name][key] = {
                    "video_name":  entry["video_name"],
                    "pred":        RISK_LABEL[entry["pred"]],
                    "label":       RISK_LABEL[entry["label"]],
                    "uncertainty": round(entry["uncertainty"], 4),
                    "n_objects":   len(entry["boxes"]),
                }
    meta_path = out_dir / "qualitative_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata → {meta_path}")


if __name__ == "__main__":
    main()
