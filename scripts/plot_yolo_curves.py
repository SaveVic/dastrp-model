"""
scripts/plot_yolo_curves.py
============================
Generate figure training curves mAP@50 dan mAP@50-95 untuk
empat model YOLO (v8n, v8s, v11n, v11s) dari hasil training
yang tersimpan di folder results/yolo/{model}/results.csv.

Output: fig_yolo_curves.pdf dan fig_yolo_curves.png
        siap dipakai sebagai fig:yolo_curves di paper.

Cara pakai:
  python scripts/plot_yolo_curves.py \
    --results-dir "runs/yolo" \
    --out-dir     "runs/yolo/figures"
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

matplotlib.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif"],
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "lines.linewidth":  1.4,
    "axes.linewidth":   0.8,
    "grid.linewidth":   0.5,
    "grid.alpha":       0.4,
    "figure.dpi":       300,
})

# ─────────────────────────────────────────────
# Konfigurasi model
# ─────────────────────────────────────────────

MODELS = [
    {"folder": "v8n",  "label": "YOLOv8n",  "color": "#185FA5", "ls": "-"},
    {"folder": "v8s",  "label": "YOLOv8s",  "color": "#BA7517", "ls": "--"},
    {"folder": "v11n", "label": "YOLO11n",  "color": "#1D9E75", "ls": "-"},
    {"folder": "v11s", "label": "YOLO11s",  "color": "#A32D2D", "ls": "--"},
]

# Kolom yang dicari di results.csv — Ultralytics kadang berubah nama antar versi
MAP50_COLS   = ["metrics/mAP50(B)",    "metrics/mAP_50",    "mAP50",    "mAP_0.5"]
MAP5095_COLS = ["metrics/mAP50-95(B)", "metrics/mAP50-95",  "mAP50-95", "mAP_0.5:0.95"]
EPOCH_COLS   = ["epoch", "Epoch"]


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_results(model_dir):
    for fname in ["results.csv", "results.txt"]:
        path = model_dir / fname
        if path.exists():
            try:
                df = pd.read_csv(path)
                df.columns = [c.strip() for c in df.columns]
                return df
            except Exception as e:
                print(f"  ⚠ Gagal baca {path}: {e}")
    return None


def smooth(values, window=3):
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    padded = np.pad(values, (window//2, window//2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(values)]


def plot_yolo_curves(results_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8),
                              constrained_layout=True)
    ax_50, ax_5095 = axes
    loaded_any = False

    for m in MODELS:
        model_dir = results_dir / m["folder"]
        df = load_results(model_dir)
        if df is None:
            print(f"  ⚠ results.csv tidak ditemukan: {model_dir}")
            continue

        epoch_col = find_col(df, EPOCH_COLS)
        epochs = (df[epoch_col].values if epoch_col
                  else np.arange(1, len(df) + 1))

        col50 = find_col(df, MAP50_COLS)
        if col50 is None:
            print(f"  ⚠ Kolom mAP50 tidak ditemukan di {m['folder']}: "
                  f"{list(df.columns)}")
            continue
        map50 = df[col50].values.astype(float)

        col5095 = find_col(df, MAP5095_COLS)
        map5095 = (df[col5095].values.astype(float)
                   if col5095 is not None else None)

        kw = {"color": m["color"], "linestyle": m["ls"],
              "label": m["label"], "zorder": 3}

        # Garis halus + raw transparan di belakang
        ax_50.plot(epochs, smooth(map50), **kw)
        ax_50.plot(epochs, map50, color=m["color"],
                   alpha=0.18, linestyle=m["ls"], linewidth=0.7, zorder=2)

        if map5095 is not None:
            ax_5095.plot(epochs, smooth(map5095), **kw)
            ax_5095.plot(epochs, map5095, color=m["color"],
                         alpha=0.18, linestyle=m["ls"], linewidth=0.7, zorder=2)

        # Tandai best epoch
        best_ep  = epochs[np.argmax(map50)]
        best_val = np.max(map50)
        ax_50.scatter([best_ep], [best_val], color=m["color"],
                      marker="*", s=60, zorder=5)
        if map5095 is not None:
            ax_5095.scatter([epochs[np.argmax(map5095)]],
                            [np.max(map5095)],
                            color=m["color"], marker="*", s=60, zorder=5)

        loaded_any = True
        print(f"  ✓ {m['label']:<10}: {len(epochs)} epoch, "
              f"best mAP50={best_val:.4f} @ ep{int(best_ep)}")

    if not loaded_any:
        print("\nTidak ada data yang berhasil dimuat.")
        return

    for ax, title, ylabel in [
        (ax_50,   "mAP@50",    "mAP@50"),
        (ax_5095, "mAP@50-95", "mAP@50-95"),
    ]:
        ax.set_title(title, pad=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=6))
        ax.spines[["top", "right"]].set_visible(False)

    # Legend — satu lokasi, duplikat dihilangkan
    handles, labels = ax_50.get_legend_handles_labels()
    seen = set()
    h_u, l_u = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            h_u.append(h)
            l_u.append(l)
    ax_50.legend(h_u, l_u, loc="lower right",
                 framealpha=0.85, edgecolor="0.8")

    fig.text(0.5, -0.03,
             r"$\bigstar$ = best epoch per model",
             ha="center", fontsize=7, color="0.5")

    for ext in ["pdf", "png"]:
        out_path = out_dir / f"fig_yolo_curves.{ext}"
        fig.savefig(str(out_path), bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
        print(f"  Disimpan → {out_path}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot YOLO training curves dari results.csv"
    )
    parser.add_argument("--results-dir", required=True,
                        help="Folder berisi v8n/, v8s/, v11n/, v11s/")
    parser.add_argument("--out-dir", default=None,
                        help="Folder output (default: {results-dir}/figures)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = (Path(args.out_dir) if args.out_dir
                   else results_dir / "figures")

    print(f"Results dir : {results_dir}")
    print(f"Output dir  : {out_dir}\n")
    plot_yolo_curves(results_dir, out_dir)


if __name__ == "__main__":
    main()
