"""
scripts/extract_sota_data.py
==============================
Ekstrak dan konsolidasikan hasil evaluasi test set untuk
semua model baseline SOTA (B1–B5) dan DASTRP-Net (Exp3)
ke satu file JSON terstruktur.

Tidak menghasilkan LaTeX atau narasi — murni ekstraksi data
metrik dari result_test.json tiap folder baseline.

Asumsi struktur folder:
  {results_dir}/exp3/result_test.json              ← DASTRP-Net
  {results_dir}/baseline/b1/result_test.json       ← B1
  {results_dir}/baseline/b2/result_test.json       ← B2
  {results_dir}/baseline/b3/result_test.json       ← B3
  {results_dir}/baseline/b4/result_test.json       ← B4
  {results_dir}/baseline/b5/result_test.json       ← B5

Output:
  sota_data.json — semua metrik terkonsolidasi per model

Cara pakai:
  python scripts/extract_sota_data.py \
    --results-dir "runs" \
    --out         "runs/sota_data.json"
"""

import json
import argparse
from pathlib import Path


# ─────────────────────────────────────────────
# Definisi model
# ─────────────────────────────────────────────

MODELS = [
    {
        "id":     "B1",
        "label":  "YOLO11n + MLP (frame-only)",
        "folder": "baseline/b1",
    },
    {
        "id":     "B2",
        "label":  "YOLO11n + BiLSTM",
        "folder": "baseline/b2",
    },
    {
        "id":     "B3",
        "label":  "YOLO11n + Transformer",
        "folder": "baseline/b3",
    },
    {
        "id":     "B4",
        "label":  "Scene-Graph GCN + LSTM",
        "folder": "baseline/b4",
    },
    {
        "id":     "B5",
        "label":  "SlowFast (detection-free)",
        "folder": "baseline/b5",
    },
    {
        "id":     "DASTRP",
        "label":  "DASTRP-Net (Exp3)",
        "folder": "exp3",
    },
]


# ─────────────────────────────────────────────
# Ekstraksi metrik
# ─────────────────────────────────────────────

def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_metrics(result: dict, split: str = "test") -> dict:
    """
    Ekstrak semua metrik yang relevan untuk tabel SOTA comparison.

    Metrik yang diekstrak:
      Frame-level  : acc, macro_f1, per-class P/R/F1, domain_acc
      Per-objek    : obj_acc, obj_macro_f1, obj_per-class F1
      Uncertainty  : mean per domain
      Distribusi   : pred_dist, true_dist
    """
    f1      = result.get(f"{split}_f1", {})
    obj_f1  = result.get(f"{split}_obj_f1", {})
    dom_acc = result.get(f"{split}_domain_acc", {})
    unc     = result.get(f"{split}_unc", {})

    def safe_f1(d, key):
        return {
            "precision": d.get(key, {}).get("precision"),
            "recall":    d.get(key, {}).get("recall"),
            "f1":        d.get(key, {}).get("f1"),
        }

    return {
        # Overall
        "acc":       result.get(f"{split}_acc"),
        "macro_f1":  f1.get("macro", {}).get("f1"),

        # Per-kelas frame level
        "safe":    safe_f1(f1, "Safe"),
        "warning": safe_f1(f1, "Warning"),
        "danger":  safe_f1(f1, "Danger"),

        # Per-domain accuracy
        "domain_acc": {
            "UGM":      dom_acc.get("UGM"),
            "Utara":    dom_acc.get("Utara"),
            "Ringroad": dom_acc.get("Ringroad"),
        },

        # Per-objek risk
        "obj_acc":      result.get(f"{split}_obj_risk_acc"),
        "obj_macro_f1": obj_f1.get("macro", {}).get("f1"),
        "obj_safe":     safe_f1(obj_f1, "Safe"),
        "obj_warning":  safe_f1(obj_f1, "Warning"),
        "obj_danger":   safe_f1(obj_f1, "Danger"),

        # Uncertainty
        "mean_unc": {
            "UGM":      unc.get("UGM"),
            "Utara":    unc.get("Utara"),
            "Ringroad": unc.get("Ringroad"),
            "overall":  (sum(v for v in unc.values() if v is not None)
                         / max(sum(1 for v in unc.values() if v is not None), 1)
                         if unc else None),
        },

        # Distribusi prediksi
        "pred_dist": result.get(f"{split}_pred_dist", {}),
        "true_dist": result.get(f"{split}_true_dist", {}),

        # Confusion matrix
        "confusion": result.get(f"{split}_confusion"),
    }


def extract_sota_data(results_dir: Path) -> dict:
    """
    Kumpulkan metrik semua model ke satu dict.

    Returns:
      {
        "B1": {...metrics...} atau None jika belum ada,
        "B2": {...},
        ...
        "DASTRP": {...},
        "missing": ["b3", "b5"],
      }
    """
    data    = {}
    missing = []

    for m in MODELS:
        model_id  = m["id"]
        folder    = results_dir / m["folder"]
        json_path = folder / "result_test.json"

        if not json_path.exists():
            print(f"  ⚠ {model_id:<8} ({m['label']}): "
                  f"result_test.json tidak ditemukan")
            missing.append(m["folder"])
            data[model_id] = None
            continue

        result  = load_json(json_path)
        metrics = extract_metrics(result)
        data[model_id] = {
            "label":   m["label"],
            "metrics": metrics,
        }

        acc   = metrics["acc"]   or 0
        macro = metrics["macro_f1"] or 0
        warn  = metrics["warning"]["f1"] or 0
        dang  = metrics["danger"]["f1"]  or 0
        print(f"  ✓ {model_id:<8} ({m['label']:<35}): "
              f"acc={acc:.2f}%  macro={macro:.2f}%  "
              f"W={warn:.2f}%  D={dang:.2f}%")

    data["missing"] = missing
    return data


# ─────────────────────────────────────────────
# Delta terhadap DASTRP-Net
# ─────────────────────────────────────────────

def compute_deltas(data: dict) -> dict:
    """
    Hitung delta setiap baseline terhadap DASTRP-Net.
    Positif = DASTRP lebih baik.
    """
    dastrp = data.get("DASTRP")
    if dastrp is None or dastrp["metrics"]["macro_f1"] is None:
        return {}

    base_macro = dastrp["metrics"]["macro_f1"]
    base_warn  = dastrp["metrics"]["warning"]["f1"]
    base_dang  = dastrp["metrics"]["danger"]["f1"]
    base_acc   = dastrp["metrics"]["acc"]

    deltas = {}
    for m in MODELS:
        mid = m["id"]
        if mid == "DASTRP" or data.get(mid) is None:
            continue
        bm = data[mid]["metrics"]
        deltas[mid] = {
            "delta_macro": (round(base_macro - (bm["macro_f1"] or 0), 2)
                            if bm["macro_f1"] is not None else None),
            "delta_warn":  (round(base_warn - (bm["warning"]["f1"] or 0), 2)
                            if bm["warning"]["f1"] is not None else None),
            "delta_dang":  (round(base_dang - (bm["danger"]["f1"] or 0), 2)
                            if bm["danger"]["f1"] is not None else None),
            "delta_acc":   (round(base_acc - (bm["acc"] or 0), 2)
                            if bm["acc"] is not None else None),
        }

    return deltas


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ekstrak data hasil comparison SOTA ke JSON"
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Folder hasil eksperimen (berisi exp3/ dan baseline/)"
    )
    parser.add_argument(
        "--out", default=None,
        help="Path output JSON "
             "(default: {results_dir}/sota_data.json)"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_path    = (Path(args.out) if args.out
                   else results_dir / "sota_data.json")

    print(f"Results dir : {results_dir}\n")

    # Ekstrak semua metrik
    data = extract_sota_data(results_dir)

    # Hitung delta terhadap DASTRP-Net
    deltas = compute_deltas(data)
    if deltas:
        print(f"\nΔ vs DASTRP-Net (positif = DASTRP lebih baik):")
        for mid, d in deltas.items():
            print(f"  {mid:<8}: ΔMacro={d['delta_macro']:+.2f}  "
                  f"ΔWarn={d['delta_warn']:+.2f}  "
                  f"ΔDang={d['delta_dang']:+.2f}")
        data["delta_vs_dastrp"] = deltas

    # Simpan
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nData tersimpan → {out_path}")
    if data["missing"]:
        print(f"Belum ada hasil  : {data['missing']}")
        print(f"Jalankan evaluate.py untuk folder di atas, "
              f"lalu jalankan script ini kembali.")


if __name__ == "__main__":
    main()
