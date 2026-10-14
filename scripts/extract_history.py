"""
scripts/extract_history.py
===========================
Rekonstruksi history.json dari TensorBoard event files.

Berguna ketika training terputus sebelum history.json sempat dibuat,
tapi TensorBoard sudah menyimpan semua scalar metrics.

Usage:
  python scripts/extract_history.py \
    --logdir "runs/exp1/tensorboard" \
    --out    "runs/exp1/history.json"

Output:
  history.json dengan format sama seperti yang dihasilkan train.py
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict


def extract_from_tensorboard(logdir: str) -> list:
    """
    Baca semua scalar dari TensorBoard event files.
    Return list of dict per epoch, format kompatibel dengan history.json.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator \
            import EventAccumulator
    except ImportError:
        raise ImportError(
            "tensorboard tidak terinstall.\n"
            "Install dengan: pip install tensorboard"
        )

    logdir = Path(logdir)
    if not logdir.exists():
        raise FileNotFoundError(f"TensorBoard logdir tidak ditemukan: {logdir}")

    print(f"Loading events dari: {logdir}")

    # Load semua event files di logdir
    ea = EventAccumulator(str(logdir))
    ea.Reload()

    # Daftar semua scalar tags yang tersedia
    tags = ea.Tags().get("scalars", [])
    print(f"Tags ditemukan ({len(tags)}):")
    for tag in sorted(tags):
        events = ea.Scalars(tag)
        print(f"  {tag:<40s}: {len(events)} data points")

    if not tags:
        print("\nTidak ada scalar tags. Pastikan logdir benar.")
        return []

    # ── Kumpulkan semua nilai per step/epoch ──────────────────
    # Key: epoch (int), Value: dict metric
    epoch_data = defaultdict(dict)

    for tag in tags:
        events = ea.Scalars(tag)
        for e in events:
            # TensorBoard step = epoch dalam kasus kita
            epoch = int(e.step)
            # Bersihkan nama tag: "train/loss_total" → "loss"
            key = _clean_tag(tag)
            epoch_data[epoch][key] = round(float(e.value), 6)

    # ── Susun menjadi list terurut per epoch ──────────────────
    history = []
    for epoch in sorted(epoch_data.keys()):
        entry = {"epoch": epoch}
        entry.update(epoch_data[epoch])
        history.append(entry)

    print(f"\nTotal epoch terekam: {len(history)}")
    if history:
        print(f"Epoch pertama : {history[0]['epoch']}")
        print(f"Epoch terakhir: {history[-1]['epoch']}")
        print(f"Keys per entry: {list(history[-1].keys())}")

    return history


def _clean_tag(tag: str) -> str:
    """
    Konversi TensorBoard tag ke key history.json.
    Contoh:
      'train/loss_total'  → 'loss'
      'train/acc'         → 'acc'
      'val/acc'           → 'val_acc'
      'val/domain_acc'    → 'val_domain_acc'
      'sys/vram_peak_gb'  → 'vram_peak_gb'
    """
    # Ganti slash dengan underscore, kecuali train/ yang dihilangkan
    tag = tag.strip()
    if tag.startswith("train/"):
        key = tag[len("train/"):]
        # Singkatkan beberapa nama
        key = key.replace("loss_total", "loss")
        return key
    elif tag.startswith("val/"):
        key = tag[len("val/"):]
        return f"val_{key}"
    elif tag.startswith("sys/"):
        return tag[len("sys/"):]
    else:
        return tag.replace("/", "_")


def check_best_epoch(history: list) -> dict:
    """
    Temukan epoch dengan val_acc tertinggi.
    """
    best = {"epoch": None, "val_acc": -1}
    for entry in history:
        val_acc = entry.get("val_acc", -1)
        if val_acc > best["val_acc"]:
            best = {"epoch": entry["epoch"], "val_acc": val_acc}
    return best


def print_summary(history: list):
    """Print ringkasan training dari history."""
    if not history:
        return

    print("\n" + "=" * 60)
    print("RINGKASAN TRAINING (dari TensorBoard)")
    print("=" * 60)

    # Cari best epoch
    best = check_best_epoch(history)
    print(f"Best epoch    : {best['epoch']} "
          f"(val_acc={best['val_acc']:.2f}%)")

    # Epoch pertama dan terakhir
    first = history[0]
    last  = history[-1]
    print(f"Epoch range   : {first['epoch']} → {last['epoch']}")

    # Loss dan accuracy training
    if "loss" in last:
        print(f"\nEpoch {last['epoch']} metrics:")
        for k, v in sorted(last.items()):
            if k == "epoch":
                continue
            print(f"  {k:<35s}: {v}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Rekonstruksi history.json dari TensorBoard events"
    )
    parser.add_argument(
        "--logdir", required=True,
        help="Path ke folder tensorboard (berisi events.out.tfevents.*)"
    )
    parser.add_argument(
        "--out", default=None,
        help="Path output history.json "
             "(default: logdir/../history.json)"
    )
    parser.add_argument(
        "--print-only", action="store_true",
        help="Hanya print summary, tidak simpan file"
    )
    args = parser.parse_args()

    # ── Ekstrak ───────────────────────────────────────────────
    history = extract_from_tensorboard(args.logdir)

    if not history:
        print("Tidak ada data yang diekstrak.")
        return

    # ── Print summary ─────────────────────────────────────────
    print_summary(history)

    if args.print_only:
        return

    # ── Simpan ────────────────────────────────────────────────
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(args.logdir).parent / "history.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nHistory disimpan → {out_path}")
    print(f"Total entries   : {len(history)}")


if __name__ == "__main__":
    main()
