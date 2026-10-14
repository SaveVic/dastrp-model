"""
scripts/extract_ablation_data.py
==================================
Ekstrak data mentah hasil ablation study dari result_test.json
tiap skenario ke satu file JSON terkonsolidasi.

Tidak menghasilkan LaTeX — hanya data terstruktur yang bisa
dipakai untuk keperluan lain (visualisasi, analisis manual,
atau diolah lebih lanjut oleh script terpisah).

Asumsi struktur folder hasil:
  {results_dir}/exp3/result_test.json              ← baseline
  {results_dir}/ablation/no_gnn/result_test.json
  {results_dir}/ablation/no_cbn/result_test.json
  {results_dir}/ablation/no_fusion/result_test.json
  {results_dir}/ablation/no_edl/result_test.json
  {results_dir}/ablation/no_velocity/result_test.json
  {results_dir}/ablation/no_obj_risk/result_test.json

Cara pakai:
  python scripts/extract_ablation_data.py \
    --results-dir "runs" \
    --out         "runs/ablation_data.json"
"""

import json
import argparse
from pathlib import Path


ABLATION_SCENARIOS = [
    ("no_gnn",      "A1"),
    ("no_cbn",      "A2"),
    ("no_fusion",   "A3"),
    ("no_edl",      "A4"),
    ("no_velocity", "A5"),
    ("no_obj_risk", "A6"),
]

BASELINE_FOLDER = "exp3"


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def extract_metrics(result: dict, split: str = "test") -> dict:
    """
    Ambil metrik dari result_test.json.
    Returns None untuk key yang tidak ditemukan (bukan error).
    """
    f1     = result.get(f"{split}_f1", {})
    obj_f1 = result.get(f"{split}_obj_f1", {})

    return {
        "acc":          result.get(f"{split}_acc"),
        "macro_f1":     f1.get("macro", {}).get("f1"),
        "safe_f1":      f1.get("Safe", {}).get("f1"),
        "warn_f1":      f1.get("Warning", {}).get("f1"),
        "dang_f1":      f1.get("Danger", {}).get("f1"),
        "safe_precision": f1.get("Safe", {}).get("precision"),
        "safe_recall":    f1.get("Safe", {}).get("recall"),
        "warn_precision": f1.get("Warning", {}).get("precision"),
        "warn_recall":    f1.get("Warning", {}).get("recall"),
        "dang_precision": f1.get("Danger", {}).get("precision"),
        "dang_recall":    f1.get("Danger", {}).get("recall"),
        "obj_macro_f1": obj_f1.get("macro", {}).get("f1"),
        "obj_safe_f1":  obj_f1.get("Safe", {}).get("f1"),
        "obj_warn_f1":  obj_f1.get("Warning", {}).get("f1"),
        "obj_dang_f1":  obj_f1.get("Danger", {}).get("f1"),
    }


def extract_ablation_data(results_dir: Path) -> dict:
    """
    Kumpulkan semua metrik baseline + 6 skenario ablasi.

    Returns dict:
      {
        "baseline": {...metrics...},
        "A1": {...metrics...} atau None jika belum ada,
        "A2": {...},
        ...
        "delta": {  # delta macro_f1 terhadap baseline
          "A1": -6.35, "A2": -3.65, ...
        },
        "missing": ["no_fusion", "no_velocity"]  # skenario tanpa data
      }
    """
    data    = {}
    missing = []

    # ── Baseline ────────────────────────────────────────────
    baseline_path = results_dir / BASELINE_FOLDER / "result_test.json"
    if not baseline_path.exists():
        raise FileNotFoundError(
            f"Baseline result tidak ditemukan: {baseline_path}"
        )
    data["baseline"] = extract_metrics(load_json(baseline_path))

    # ── Setiap skenario ablasi ─────────────────────────────────
    for folder_name, ablation_id in ABLATION_SCENARIOS:
        result_path = (results_dir / "ablation" / folder_name
                       / "result_test.json")
        if result_path.exists():
            data[ablation_id] = extract_metrics(load_json(result_path))
        else:
            data[ablation_id] = None
            missing.append(folder_name)

    # ── Hitung delta macro F1 terhadap baseline ─────────────────
    base_macro = data["baseline"]["macro_f1"]
    deltas = {}
    for _, ablation_id in ABLATION_SCENARIOS:
        m = data[ablation_id]
        if m is not None and m["macro_f1"] is not None and base_macro is not None:
            deltas[ablation_id] = round(m["macro_f1"] - base_macro, 4)
        else:
            deltas[ablation_id] = None

    data["delta"]   = deltas
    data["missing"] = missing

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Ekstrak data ablation study ke JSON"
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Folder hasil eksperimen (berisi exp3/ dan ablation/)"
    )
    parser.add_argument(
        "--out", default=None,
        help="Path output JSON "
             "(default: {results_dir}/ablation_data.json)"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_path    = (Path(args.out) if args.out
                   else results_dir / "ablation_data.json")

    data = extract_ablation_data(results_dir)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Data ablasi → {out_path}")
    if data["missing"]:
        print(f"Skenario belum ada hasil: {data['missing']}")


if __name__ == "__main__":
    main()
