"""
scripts/plot_uncertainty_figure.py
=====================================
Generate fig:uncertainty — distribusi uncertainty per-clip
dipisah berdasarkan (a) correct vs incorrect prediction
dan (b) domain.

Membutuhkan result_test.json Exp3 yang sudah mengandung
key "test_unc_detail" dari evaluate.py yang sudah diperbarui.

Jika result_test.json lama (belum punya unc_detail), script
menyediakan opsi --rerun untuk menjalankan ulang evaluate.py
secara otomatis.

Output:
  fig_uncertainty.pdf / .png  — siap untuk figure* dua kolom

Cara pakai:
  python scripts/plot_uncertainty_figure.py \
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
    "lines.linewidth":  1.2,
    "axes.linewidth":   0.8,
    "grid.linewidth":   0.5,
    "grid.alpha":       0.4,
    "figure.dpi":       300,
})

DOMAIN_MAP  = {0: "UGM", 1: "Utara", 2: "Ringroad"}
RISK_LABELS = {0: "Safe", 1: "Warning", 2: "Danger"}

# Warna konsisten dengan figure lain di paper
COLOR_CORRECT   = "#185FA5"   # biru
COLOR_INCORRECT = "#A32D2D"   # merah
DOMAIN_COLORS   = {
    "UGM":      "#BA7517",
    "Utara":    "#1D9E75",
    "Ringroad": "#185FA5",
}


def load_unc_detail(exp3_dir: Path, split: str = "test") -> dict | None:
    """
    Baca unc_detail dari result_{split}.json.
    Returns dict atau None jika key tidak ditemukan.
    """
    result_path = exp3_dir / f"result_{split}.json"
    if not result_path.exists():
        print(f"  ⚠ Tidak ditemukan: {result_path}")
        return None

    with open(result_path) as f:
        result = json.load(f)

    key = f"{split}_unc_detail"
    if key not in result:
        print(f"  ⚠ Key '{key}' tidak ada di result_{split}.json")
        print(f"    Jalankan ulang evaluate.py dengan versi terbaru "
              f"(yang sudah menyimpan unc_detail) terlebih dahulu.")
        return None

    detail = result[key]
    return {
        "uncertainty": np.array(detail["uncertainty"]),
        "label":       np.array(detail["label"]),
        "domain":      np.array(detail["domain"]),
        "pred":        np.array(detail["pred"]),
    }


def kde_smooth(values: np.ndarray, n_points: int = 300,
               bandwidth: float = 0.02) -> tuple:
    """
    KDE manual menggunakan Gaussian kernel.
    Lebih ringan dari scipy.stats.gaussian_kde.
    """
    x = np.linspace(0, 1, n_points)
    density = np.zeros(n_points)
    for v in values:
        density += np.exp(-0.5 * ((x - v) / bandwidth) ** 2)
    density /= (len(values) * bandwidth * np.sqrt(2 * np.pi))
    return x, density


def plot_uncertainty_figure(exp3_dir: Path, out_dir: Path,
                             split: str = "test"):
    """
    Dua panel:
      (a) KDE: correct vs incorrect prediction
      (b) KDE: per domain (UGM / Utara / Ringroad)
    """
    detail = load_unc_detail(exp3_dir, split)
    if detail is None:
        return

    unc     = detail["uncertainty"]
    labels  = detail["label"]
    domains = detail["domain"]
    preds   = detail["pred"]
    correct = (preds == labels)

    n_total     = len(unc)
    n_correct   = correct.sum()
    n_incorrect = (~correct).sum()
    print(f"  Total clips  : {n_total}")
    print(f"  Correct      : {n_correct} ({n_correct/n_total*100:.1f}%)")
    print(f"  Incorrect    : {n_incorrect} ({n_incorrect/n_total*100:.1f}%)")
    print(f"  Mean unc (C) : {unc[correct].mean():.4f}")
    print(f"  Mean unc (I) : {unc[~correct].mean():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6),
                              constrained_layout=True)

    # ── Panel (a): Correct vs Incorrect ───────────────────────
    ax = axes[0]
    bw = 0.025

    for mask, label, color, ls, n in [
        (correct,  f"Correct ($n={n_correct}$)",
         COLOR_CORRECT,   "-",  n_correct),
        (~correct, f"Incorrect ($n={n_incorrect}$)",
         COLOR_INCORRECT, "--", n_incorrect),
    ]:
        if mask.sum() == 0:
            continue
        vals = unc[mask]
        x, y = kde_smooth(vals, bandwidth=bw)
        ax.fill_between(x, y, alpha=0.12, color=color)
        ax.plot(x, y, color=color, linestyle=ls, linewidth=1.4,
                label=label)
        # Rug plot (tanda kecil di sumbu x)
        ax.plot(vals, np.full_like(vals, -0.05), "|",
                color=color, alpha=0.3, markersize=3, markeredgewidth=0.6)

    ax.axvline(x=unc[correct].mean(),  color=COLOR_CORRECT,
               linestyle=":", linewidth=1.0, alpha=0.7)
    ax.axvline(x=unc[~correct].mean(), color=COLOR_INCORRECT,
               linestyle=":", linewidth=1.0, alpha=0.7)

    ax.set_xlabel("Uncertainty Score $u$")
    ax.set_ylabel("Density")
    ax.set_title("(a) Correct vs. Incorrect Prediction", pad=4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=-0.3)
    ax.legend(loc="upper center", framealpha=0.85, edgecolor="0.8")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="x")

    # ── Panel (b): Per domain ─────────────────────────────────
    ax = axes[1]
    domain_ids = sorted(DOMAIN_MAP.keys())

    for d_id in domain_ids:
        d_name = DOMAIN_MAP[d_id]
        mask   = (domains == d_id)
        if mask.sum() == 0:
            continue
        vals   = unc[mask]
        color  = DOMAIN_COLORS[d_name]
        n_dom  = mask.sum()
        mean_u = vals.mean()

        x, y = kde_smooth(vals, bandwidth=bw)
        ax.fill_between(x, y, alpha=0.12, color=color)
        ax.plot(x, y, color=color, linewidth=1.4,
                label=f"{d_name} ($n={n_dom}$, $\\bar{{u}}={mean_u:.3f}$)")
        ax.plot(vals, np.full_like(vals, -0.05), "|",
                color=color, alpha=0.25, markersize=3, markeredgewidth=0.6)
        ax.axvline(x=mean_u, color=color, linestyle=":",
                   linewidth=1.0, alpha=0.7)

    ax.set_xlabel("Uncertainty Score $u$")
    ax.set_ylabel("Density")
    ax.set_title("(b) Per-Domain Distribution", pad=4)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=-0.3)
    ax.legend(loc="upper center", framealpha=0.85, edgecolor="0.8",
              fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="x")

    fig.text(0.5, -0.04,
             "Dashed vertical lines indicate group mean uncertainty. "
             "Rug marks show individual clip uncertainty values.",
             ha="center", fontsize=7, color="0.5")

    # ── Simpan ───────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "png"]:
        out_path = out_dir / f"fig_uncertainty.{ext}"
        fig.savefig(str(out_path), bbox_inches="tight",
                    dpi=300 if ext == "png" else None)
        print(f"  Disimpan → {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate fig:uncertainty dari result_test.json"
    )
    parser.add_argument("--exp3-dir", required=True,
                        help="Folder Exp3 (berisi result_test.json)")
    parser.add_argument("--out-dir", default=None,
                        help="Folder output (default: {exp3-dir}/figures)")
    parser.add_argument("--split", default="test",
                        choices=["val", "test"])
    args = parser.parse_args()

    exp3_dir = Path(args.exp3_dir)
    out_dir  = (Path(args.out_dir) if args.out_dir
                else exp3_dir / "figures")

    print(f"Exp3 dir : {exp3_dir}")
    print(f"Out dir  : {out_dir}\n")
    plot_uncertainty_figure(exp3_dir, out_dir, args.split)


if __name__ == "__main__":
    main()
