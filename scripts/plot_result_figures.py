"""
scripts/plot_result_figures.py
================================
Generate dua figure untuk section result:

  1. fig:confusion  — confusion matrix heatmap Exp3 test set
  2. fig:training_curve — val macro F1, Warning F1, Danger F1
                          per epoch untuk Exp2 dan Exp3

Input yang dibutuhkan:
  - result_test.json  dari evaluate.py (untuk confusion matrix)
  - history.json      dari train.py    (untuk training curve)

Cara pakai:
  python scripts/plot_result_figures.py \
    --exp2-dir ".../results/dastrp/exp2" \
    --exp3-dir ".../results/dastrp/exp3" \
    --out-dir  ".../results/dastrp/figures"
"""

import json
import argparse
import numpy as np
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

RISK_LABELS = ["Safe", "Warning", "Danger"]


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_fig(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        p = out_dir / f"{name}.{ext}"
        fig.savefig(str(p), bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
        print(f"  Disimpan → {p}")
    plt.close(fig)


# ─────────────────────────────────────────────
# 1. Confusion Matrix
# ─────────────────────────────────────────────

def plot_confusion_matrix(exp3_dir: Path, out_dir: Path):
    """
    Heatmap confusion matrix dari result_test.json Exp3.
    Setiap sel menampilkan jumlah absolut + persentase per baris (recall basis).
    Sumbu: x = Predicted, y = True.
    """
    result_path = exp3_dir / "result_test.json"
    if not result_path.exists():
        print(f"  ⚠ Tidak ditemukan: {result_path}")
        return

    result = load_json(result_path)
    cm_key = next((k for k in result if k.endswith("_confusion")), None)
    if cm_key is None:
        print("  ⚠ Key confusion matrix tidak ditemukan di result_test.json")
        return

    cm = np.array(result[cm_key])   # (3, 3)
    n  = cm.shape[0]

    # Normalisasi per baris (recall-based) untuk warna, nilai absolut di sel
    row_sum = cm.sum(axis=1, keepdims=True).astype(float)
    cm_norm = np.where(row_sum > 0, cm / row_sum, 0)

    fig, ax = plt.subplots(figsize=(3.5, 3.0), constrained_layout=True)

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    # Colorbar kompak
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Anotasi setiap sel
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            count   = int(cm[i, j])
            pct     = cm_norm[i, j] * 100
            color   = "white" if cm_norm[i, j] > thresh else "black"
            txt     = f"{count}\n({pct:.1f}%)"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=7.5, color=color,
                    fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(RISK_LABELS)
    ax.set_yticklabels(RISK_LABELS)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (Exp3 Test Set)", pad=6)

    # Garis pemisah diagonal tipis untuk memudahkan mata
    for i in range(n):
        ax.add_patch(plt.Rectangle(
            (i - 0.5, i - 0.5), 1, 1,
            fill=False, edgecolor="white", linewidth=1.5, zorder=3
        ))

    save_fig(fig, out_dir, "fig_confusion")


# ─────────────────────────────────────────────
# 2. Training Curve
# ─────────────────────────────────────────────

def smooth(values, window=5):
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window//2, window//2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[:len(arr)]


def load_tensorboard_metrics(exp_dir: Path) -> dict | None:
    """
    Baca metrik training dari TensorBoard event files.
    Mencari folder tensorboard/ di dalam exp_dir secara otomatis.

    Returns dict dengan key epoch, macro, warn, danger
    atau None jika tidak ditemukan / gagal.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator \
            import EventAccumulator
    except ImportError:
        print("  ⚠ tensorboard tidak terinstall: pip install tensorboard")
        return None

    # Cari folder tensorboard
    tb_candidates = [
        exp_dir / "tensorboard",
        exp_dir / "tb",
        exp_dir / "logs",
        exp_dir,   # mungkin langsung di root
    ]
    tb_dir = next((d for d in tb_candidates
                   if d.exists() and any(d.glob("events.out.tfevents.*"))),
                  None)

    if tb_dir is None:
        print(f"  ⚠ Folder TensorBoard tidak ditemukan di: {exp_dir}")
        return None

    print(f"  Membaca TensorBoard dari: {tb_dir}")
    ea = EventAccumulator(str(tb_dir))
    ea.Reload()

    available = ea.Tags().get("scalars", [])

    # Nama tag yang mungkin berbeda antar versi train.py
    MACRO_TAGS  = ["val/macro_f1", "val/macro_F1",
                   "val/f1_macro", "val_macro_f1"]
    WARN_TAGS   = ["val/acc_Warning", "val/f1_Warning",
                   "val/Warning_f1",  "val_acc_Warning"]
    DANGER_TAGS = ["val/acc_Danger",  "val/f1_Danger",
                   "val/Danger_f1",   "val_acc_Danger"]
    SAFE_TAGS   = ["val/acc_Safe",    "val/f1_Safe",
                   "val/Safe_f1",     "val_acc_Safe"]

    def find_tag(candidates):
        for t in candidates:
            if t in available:
                return t
        return None

    macro_tag  = find_tag(MACRO_TAGS)
    warn_tag   = find_tag(WARN_TAGS)
    danger_tag = find_tag(DANGER_TAGS)
    safe_tag   = find_tag(SAFE_TAGS)

    if macro_tag is None and warn_tag is None:
        print(f"  ⚠ Tag F1 tidak ditemukan. Tags tersedia: {available[:10]}")
        return None

    if macro_tag is None:
        # Fallback: hitung macro sebagai mean(Safe, Warning, Danger) per-class acc
        if safe_tag and warn_tag and danger_tag:
            print(f"  Macro F1 tag tidak ditemukan → "
                  f"dihitung sebagai mean(Safe, Warning, Danger) per-class acc")
            macro_tag = "__computed__"
        else:
            print(f"  ⚠ Macro F1 tag tidak ditemukan di antara kandidat: "
                  f"{MACRO_TAGS}")
            print(f"  Tags tersedia (semua): {sorted(available)}")
            print(f"  → Panel Macro F1 akan kosong.")
    else:
        print(f"  Macro tag ditemukan: {macro_tag}")

    def read_tag(tag):
        if tag is None or tag not in available:
            return {}, []
        events = ea.Scalars(tag)
        return (
            {int(e.step): float(e.value) for e in events},
            [int(e.step) for e in events],
        )

    macro_d,  macro_eps  = read_tag(macro_tag  if macro_tag != "__computed__" else None)
    warn_d,   warn_eps   = read_tag(warn_tag)
    danger_d, danger_eps = read_tag(danger_tag)
    safe_d,   safe_eps   = read_tag(safe_tag)

    # Hitung macro dari per-class jika tidak ada tag langsung
    if macro_tag == "__computed__":
        all_eps_raw = sorted(set(warn_eps) | set(danger_eps) | set(safe_eps))
        for ep in all_eps_raw:
            s = safe_d.get(ep,   np.nan)
            w = warn_d.get(ep,   np.nan)
            d = danger_d.get(ep, np.nan)
            vals = [v for v in [s, w, d] if not np.isnan(v)]
            macro_d[ep]  = np.mean(vals) if vals else np.nan
            macro_eps.append(ep)

    # Epoch yang tersedia di minimal satu tag
    all_eps = sorted(set(macro_eps) | set(warn_eps) | set(danger_eps))
    if not all_eps:
        print("  ⚠ Tidak ada data epoch ditemukan di TensorBoard")
        return None

    metrics = {
        "epoch":  all_eps,
        "macro":  [macro_d.get(ep, np.nan)  for ep in all_eps],
        "warn":   [warn_d.get(ep, np.nan)   for ep in all_eps],
        "danger": [danger_d.get(ep, np.nan) for ep in all_eps],
    }

    # ── Auto-scale: deteksi jika nilai disimpan 0-1, konversi ke 0-100 ──
    # train.py bisa menyimpan macro_f1 sebagai 0-1 (dari dict hasil evaluate)
    # tapi Warning/Danger acc disimpan langsung sebagai 0-100
    for key in ["macro", "warn", "danger"]:
        vals = [v for v in metrics[key] if not np.isnan(v) and v > 0]
        if vals and max(vals) <= 1.5:
            # Semua nilai di bawah 1.5 → kemungkinan skala 0-1
            print(f"  Auto-scale '{key}': terdeteksi skala 0-1, "
                  f"dikali 100 (max raw={max(vals):.4f})")
            metrics[key] = [v * 100 if not np.isnan(v) else np.nan
                            for v in metrics[key]]

    valid = [not (np.isnan(m) and np.isnan(w) and np.isnan(d))
             for m, w, d in zip(metrics["macro"],
                                metrics["warn"],
                                metrics["danger"])]
    if not any(valid):
        return None

    print(f"  Tags   : macro={macro_tag}, warn={warn_tag}, "
          f"danger={danger_tag}")
    print(f"  Epochs : {len(all_eps)} "
          f"({all_eps[0]}–{all_eps[-1]})")

    return metrics


def plot_training_curves(
    exp2_dir: Path,
    exp3_dir: Path,
    out_dir:  Path,
):
    """
    Tiga panel: Macro F1, Warning F1, Danger F1 per epoch.
    Data dibaca dari TensorBoard event files.
    """
    histories = {}
    for label, exp_dir in [("Exp2 (original split)", exp2_dir),
                             ("Exp3 (balanced split)",  exp3_dir)]:
        print(f"\n  [{label}]")
        metrics = load_tensorboard_metrics(exp_dir)
        if metrics is None:
            print(f"  → Dilewati")
            continue

        # Bersihkan NaN
        for key in ["macro", "warn", "danger"]:
            arr = np.array(metrics[key], dtype=float)
            # Forward-fill NaN
            for i in range(1, len(arr)):
                if np.isnan(arr[i]):
                    arr[i] = arr[i-1]
            # Jika masih NaN di awal, isi 0
            arr = np.where(np.isnan(arr), 0, arr)
            metrics[key] = arr

        valid_macro = [v for v in metrics["macro"] if v > 0]
        if valid_macro:
            print(f"  Best macro F1 = {max(valid_macro):.2f}%")
        histories[label] = metrics

    if not histories:
        print("\n  Tidak ada data training curve yang bisa dimuat.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5),
                              constrained_layout=True)

    style = {
        "Exp2 (original split)": {
            "color": "#73726c", "ls": "--", "lw": 1.3, "alpha_raw": 0.15
        },
        "Exp3 (balanced split)": {
            "color": "#185FA5", "ls": "-",  "lw": 1.5, "alpha_raw": 0.15
        },
    }

    metrics_to_plot = [
        ("macro",  "Macro F1 (%)",   "Macro F1"),
        ("warn",   "Warning F1 (%)", "Warning F1"),
        ("danger", "Danger F1 (%)",  "Danger F1"),
    ]

    for ax, (key, ylabel, title) in zip(axes, metrics_to_plot):
        for label, m in histories.items():
            s  = style[label]
            ep = np.array(m["epoch"])
            v  = np.array(m[key], dtype=float)

            ax.plot(ep, v, color=s["color"], alpha=s["alpha_raw"],
                    linestyle=s["ls"], linewidth=0.7, zorder=2)
            ax.plot(ep, smooth(v), color=s["color"], linestyle=s["ls"],
                    linewidth=s["lw"], label=label, zorder=3)

            best_idx = np.nanargmax(v)
            ax.scatter([ep[best_idx]], [v[best_idx]],
                       color=s["color"], marker="*", s=55, zorder=5)

        ax.set_title(title, pad=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=5))
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)

    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    h_u, l_u = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            h_u.append(h)
            l_u.append(l)
    axes[0].legend(h_u, l_u, loc="lower right",
                   framealpha=0.85, edgecolor="0.8",
                   fontsize=7.5)

    fig.text(0.5, -0.04,
             r"$\bigstar$ = best epoch  |  Shaded = raw values  |  "
             r"Line = smoothed ($w=5$)",
             ha="center", fontsize=7, color="0.5")

    save_fig(fig, out_dir, "fig_training_curve")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate fig:confusion dan fig:training_curve"
    )
    parser.add_argument("--exp2-dir", required=True,
                        help="Folder hasil Exp2 (berisi result_test.json + history.json)")
    parser.add_argument("--exp3-dir", required=True,
                        help="Folder hasil Exp3 (berisi result_test.json + history.json)")
    parser.add_argument("--out-dir",  default=None,
                        help="Folder output figure "
                             "(default: {exp3_dir}/figures)")
    parser.add_argument("--skip-confusion",    action="store_true")
    parser.add_argument("--skip-training-curve", action="store_true")
    args = parser.parse_args()

    exp2_dir = Path(args.exp2_dir)
    exp3_dir = Path(args.exp3_dir)
    out_dir  = (Path(args.out_dir) if args.out_dir
                else exp3_dir / "figures")

    print(f"Exp2 dir : {exp2_dir}")
    print(f"Exp3 dir : {exp3_dir}")
    print(f"Out dir  : {out_dir}\n")

    if not args.skip_confusion:
        print("=== fig:confusion ===")
        plot_confusion_matrix(exp3_dir, out_dir)
        print()

    if not args.skip_training_curve:
        print("=== fig:training_curve ===")
        plot_training_curves(exp2_dir, exp3_dir, out_dir)


if __name__ == "__main__":
    main()
