"""
scripts/evaluate_ablation_batch.py
====================================
Jalankan evaluate.py pada test set untuk semua skenario ablasi
sekaligus, tanpa perlu menjalankan satu per satu secara manual.

Mengasumsikan setiap skenario sudah selesai training dan punya
folder berisi best_model.pth + config.json (hasil train.py),
tapi belum punya result_test.json (belum dievaluasi).

Struktur folder yang diharapkan:
  {results_dir}/exp3/best_model.pth + config.json
  {results_dir}/ablation/no_gnn/best_model.pth + config.json
  {results_dir}/ablation/no_cbn/best_model.pth + config.json
  {results_dir}/ablation/no_fusion/best_model.pth + config.json
  {results_dir}/ablation/no_edl/best_model.pth + config.json
  {results_dir}/ablation/no_velocity/best_model.pth + config.json
  {results_dir}/ablation/no_obj_risk/best_model.pth + config.json

Setelah dijalankan, setiap folder akan punya result_test.json
yang dihasilkan oleh evaluate.py (lihat scripts/evaluate.py).

Cara pakai:
  python scripts/evaluate_ablation_batch.py \
    --results-dir "runs"

  # Skip skenario yang sudah punya result_test.json
  python scripts/evaluate_ablation_batch.py \
    --results-dir "runs" \
    --skip-existing
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


ABLATION_FOLDERS = [
    "no_gnn", "no_cbn", "no_fusion",
    "no_edl", "no_velocity", "no_obj_risk",
]
BASELINE_FOLDER = "exp3"


def find_config(folder: Path) -> Path | None:
    """Cari config.json di dalam folder hasil training."""
    cfg_path = folder / "config.json"
    return cfg_path if cfg_path.exists() else None


def run_evaluate(config_path: Path, split: str = "test") -> bool:
    """
    Panggil evaluate.run_test() langsung (bukan subprocess)
    agar lebih cepat — tidak perlu spawn proses Python baru
    dan tidak perlu reload CUDA context tiap kali.

    Returns True jika berhasil, False jika gagal.
    """
    from scripts.evaluate import run_test

    try:
        run_test(str(config_path), split)
        return True
    except Exception as e:
        print(f"  ✗ GAGAL: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Evaluasi test set untuk semua skenario ablasi"
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Folder hasil eksperimen (berisi exp3/ dan ablation/)"
    )
    parser.add_argument(
        "--split", default="test",
        choices=["train", "val", "test"],
        help="Split yang dievaluasi (default: test)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Lewati folder yang sudah punya result_{split}.json"
    )
    parser.add_argument(
        "--include-baseline", action="store_true", default=True,
        help="Sertakan folder exp3 (baseline) — default: True"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # ── Susun daftar folder yang perlu dievaluasi ──────────────
    targets = []   # list of (label, folder_path)

    if args.include_baseline:
        baseline_dir = results_dir / BASELINE_FOLDER
        if baseline_dir.exists():
            targets.append(("baseline (exp3)", baseline_dir))
        else:
            print(f"⚠ Folder baseline tidak ditemukan: {baseline_dir}")

    for folder_name in ABLATION_FOLDERS:
        folder_path = results_dir / "ablation" / folder_name
        if folder_path.exists():
            targets.append((folder_name, folder_path))
        else:
            print(f"⚠ Folder ablasi tidak ditemukan: {folder_path}")

    if not targets:
        print("Tidak ada folder yang ditemukan untuk dievaluasi.")
        return

    print(f"{'='*65}")
    print(f"BATCH EVALUATION — {len(targets)} skenario")
    print(f"{'='*65}\n")

    # ── Loop evaluasi ────────────────────────────────────────
    results_summary = []

    for label, folder_path in targets:
        print(f"{'─'*65}")
        print(f"[{label}] {folder_path}")
        print(f"{'─'*65}")

        # Cek apakah best_model.pth ada
        ckpt_path = folder_path / "best_model.pth"
        if not ckpt_path.exists():
            print(f"  ✗ SKIP: best_model.pth tidak ditemukan "
                  f"(training belum selesai?)")
            results_summary.append((label, "SKIP — no checkpoint"))
            continue

        # Cek config.json
        cfg_path = find_config(folder_path)
        if cfg_path is None:
            print(f"  ✗ SKIP: config.json tidak ditemukan")
            results_summary.append((label, "SKIP — no config"))
            continue

        # Cek apakah sudah ada result_test.json
        result_path = folder_path / f"result_{args.split}.json"
        if args.skip_existing and result_path.exists():
            print(f"  → SKIP: result_{args.split}.json sudah ada "
                  f"(gunakan tanpa --skip-existing untuk re-evaluate)")
            results_summary.append((label, "SKIP — already evaluated"))
            continue

        # Jalankan evaluasi
        print(f"  Evaluating split='{args.split}'...")
        success = run_evaluate(cfg_path, args.split)

        if success:
            results_summary.append((label, "OK"))
            print(f"  ✓ Selesai — result_{args.split}.json tersimpan")
        else:
            results_summary.append((label, "FAILED"))

        print()

    # ── Ringkasan akhir ──────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"RINGKASAN BATCH EVALUATION")
    print(f"{'='*65}")
    for label, status in results_summary:
        symbol = "✓" if status == "OK" else "✗" if "FAILED" in status else "→"
        print(f"  {symbol} {label:<20s}: {status}")

    n_ok = sum(1 for _, s in results_summary if s == "OK")
    print(f"\n  Total: {n_ok}/{len(results_summary)} berhasil dievaluasi")

    failed = [label for label, s in results_summary
              if s == "FAILED" or "SKIP" in s and "already" not in s]
    if failed:
        print(f"\n  Perlu perhatian: {failed}")
        print(f"  Jalankan training untuk skenario ini terlebih dahulu, "
              f"lalu jalankan ulang script ini.")


if __name__ == "__main__":
    main()
