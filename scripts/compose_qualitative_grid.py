"""
scripts/compose_qualitative_grid.py
=====================================
Gabungkan 6 panel gambar kualitatif yang SUDAH ADA (hasil generate
sebelumnya) menjadi satu grid landscape untuk figure* paper.

Tidak menjalankan model atau memuat dataset — murni image composition
dari file PNG yang sudah ada di disk.

Input yang diharapkan (default nama dari generate_qualitative_figure.py):
  {in_dir}/ugm_correct.png        {in_dir}/ugm_incorrect.png
  {in_dir}/utara_correct.png      {in_dir}/utara_incorrect.png
  {in_dir}/ringroad_correct.png   {in_dir}/ringroad_incorrect.png

Layout output: LANDSCAPE — 2 baris (correct/incorrect) x 3 kolom (domain)
  Rasio lebar:tinggi ~1.6:1, sesuai untuk figure* dua kolom IEEE.

Cara pakai:
  python scripts/compose_qualitative_grid.py \
    --in-dir  ".../exp3/qualitative" \
    --out     ".../exp3/qualitative/fig_qualitative.png"
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


DOMAIN_ORDER = ["UGM", "Utara", "Ringroad"]
DOMAIN_FILE_PREFIX = {"UGM": "ugm", "Utara": "utara", "Ringroad": "ringroad"}
ROW_KEYS  = ["correct", "incorrect"]
ROW_LABEL = {"correct": "Correct Prediction", "incorrect": "Incorrect Prediction"}
ROW_COLOR = {"correct": (80, 200, 50), "incorrect": (60, 60, 255)}


def load_panel(path: Path, panel_size: int) -> np.ndarray:
    """Baca panel PNG, resize ke ukuran seragam. Placeholder jika tidak ada."""
    if not path.exists():
        panel = np.full((panel_size, panel_size, 3), 40, dtype=np.uint8)
        cv2.putText(panel, "N/A", (panel_size//2 - 30, panel_size//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2)
        print(f"  ⚠ Tidak ditemukan: {path} — pakai placeholder")
        return panel

    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Gagal membaca gambar: {path}")
    return cv2.resize(img, (panel_size, panel_size))


def compose_grid(
    in_dir:      Path,
    out_path:    Path,
    panel_size:  int = 360,
    label_h:     int = 32,
    row_label_w: int = 140,
):
    """
    Susun grid landscape: baris=[correct, incorrect], kolom=domain.
    Hanya membaca file panel yang sudah ada, tidak meng-generate apa pun.
    """
    print(f"Membaca panel dari: {in_dir}\n")

    # ── Header kolom (nama domain) ─────────────────────────────
    col_header = np.full((label_h, panel_size * len(DOMAIN_ORDER), 3),
                         30, dtype=np.uint8)
    for i, d_name in enumerate(DOMAIN_ORDER):
        x = i * panel_size + panel_size // 2 - 30
        cv2.putText(col_header, d_name, (x, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ── Susun tiap baris (correct / incorrect) ──────────────────
    rows = []
    for key in ROW_KEYS:
        panels = []
        for d_name in DOMAIN_ORDER:
            prefix = DOMAIN_FILE_PREFIX[d_name]
            panel_path = in_dir / f"{prefix}_{key}.png"
            print(f"  [{d_name} - {key}] {panel_path.name}")
            panels.append(load_panel(panel_path, panel_size))

        row_img = np.hstack(panels)

        # Label baris (Correct/Incorrect) — teks vertikal via rotasi
        label_canvas = np.full((row_label_w, panel_size, 3),
                               30, dtype=np.uint8)
        cv2.putText(label_canvas, ROW_LABEL[key], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, ROW_COLOR[key], 2)
        label_rot = cv2.rotate(label_canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)

        rows.append(np.hstack([label_rot, row_img]))

    # ── Gabungkan semua ──────────────────────────────────────────
    header_padded = np.hstack([
        np.full((label_h, row_label_w, 3), 30, dtype=np.uint8),
        col_header,
    ])
    grid = np.vstack([header_padded] + rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)

    print(f"\nGrid tersimpan → {out_path}")
    print(f"  Resolusi : {grid.shape[1]}×{grid.shape[0]} px")
    print(f"  Rasio    : {grid.shape[1]/grid.shape[0]:.2f}:1 (landscape)")


def main():
    parser = argparse.ArgumentParser(
        description="Gabungkan panel kualitatif yang sudah ada menjadi grid landscape"
    )
    parser.add_argument(
        "--in-dir", required=True,
        help="Folder berisi 6 panel PNG yang sudah ada "
             "(ugm_correct.png, ugm_incorrect.png, dst)"
    )
    parser.add_argument(
        "--out", default=None,
        help="Path output grid gabungan "
             "(default: {in-dir}/fig_qualitative.png)"
    )
    parser.add_argument(
        "--panel-size", type=int, default=360,
        help="Ukuran tiap panel setelah di-resize (px persegi)"
    )
    args = parser.parse_args()

    in_dir   = Path(args.in_dir)
    out_path = (Path(args.out) if args.out
                else in_dir / "fig_qualitative.png")

    compose_grid(in_dir, out_path, panel_size=args.panel_size)


if __name__ == "__main__":
    main()
