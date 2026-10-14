"""
split_videos.py
===============
Menentukan pembagian video train/val/test dengan mempertimbangkan:
  1. Domain (ringroad, utara, ugm)
  2. Crowdedness score (low/medium/high)
  3. Distribusi risk (rasio safe/warning/danger) yang merata antar split

Input:
  --meta-dir : folder berisi file meta.json (satu per split saat ini,
               atau folder annotation JSON per video)
  --ann-dir  : folder annotations/full berisi {video}.json per video
  --out      : path output hasil split (JSON)

Output:
  split_result.json  : assignment video ke train/val/test
  split_report.txt   : laporan distribusi lengkap

Strategi:
  - Stratified split berdasarkan (domain × crowdedness_category)
  - Di dalam tiap strata, pilih video yang paling menyeimbangkan
    distribusi risk di tiap split (greedy balancing)
  - Rasio target: train=70%, val=15%, test=15%
  - Minimum 1 video per domain di val dan test
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


# ─────────────────────────────────────────────
# Konstanta
# ─────────────────────────────────────────────

RISK_MAP    = {"safe": 0, "warning": 1, "danger": 2}
RISK_NAMES  = {0: "safe", 1: "warning", 2: "danger"}
DOMAIN_MAP  = {"ugm": 0, "utara": 1, "ringroad": 2}

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Crowdedness weights (alpha, beta, gamma)
ALPHA, BETA, GAMMA = 0.4, 0.3, 0.3
N_MAX               = 9
LAMBDA_VAR          = 0.5


# ─────────────────────────────────────────────
# 1. Baca anotasi per video
# ─────────────────────────────────────────────

def load_video_stats(ann_dir: Path) -> dict:
    """
    Baca semua file {video}.json dari ann_dir.
    Hitung statistik per video:
      - domain
      - total_frames
      - frames_with_obj
      - risk distribution (safe/warning/danger) — di level objek
      - frame-level risk distribution (max risk per frame)
      - n_max_per_frame (untuk crowdedness)
    """
    stats = {}

    for ann_path in sorted(ann_dir.glob("*.json")):
        video_name = ann_path.stem
        domain     = _infer_domain(video_name)

        with open(ann_path) as f:
            ann = json.load(f)

        start = ann.get("start", 0)
        stop  = ann.get("stop",  0)
        data  = ann.get("data",  [])

        total_frames    = stop - start + 1
        frames_with_obj = len(data)

        # Object-level risk counts
        obj_risk  = defaultdict(int)   # safe/warning/danger count
        # Frame-level risk (max per frame)
        frame_risk = defaultdict(int)  # 0/1/2 count
        # Per-frame N untuk crowdedness
        n_per_frame = []

        for entry in data:
            anns = entry.get("annotations", [])
            if not anns:
                continue
            n_per_frame.append(len(anns))
            max_risk = 0
            for obj in anns:
                r = RISK_MAP.get(str(obj.get("risk", "safe")).lower(), 0)
                obj_risk[RISK_NAMES[r]] += 1
                max_risk = max(max_risk, r)
            frame_risk[max_risk] += 1

        stats[video_name] = {
            "domain":            domain,
            "total_frames":      total_frames,
            "frames_with_obj":   frames_with_obj,
            "frames_empty":      total_frames - frames_with_obj,
            "obj_safe":          obj_risk["safe"],
            "obj_warning":       obj_risk["warning"],
            "obj_danger":        obj_risk["danger"],
            "frame_safe":        frame_risk[0],
            "frame_warning":     frame_risk[1],
            "frame_danger":      frame_risk[2],
            "n_per_frame":       n_per_frame,
        }

    return stats


def _infer_domain(video_name: str) -> str:
    name = video_name.lower()
    for prefix in ["ugm", "utara", "ringroad"]:
        if name.startswith(prefix):
            return prefix
    return "unknown"


# ─────────────────────────────────────────────
# 2. Hitung crowdedness score
# ─────────────────────────────────────────────

def compute_crowdedness(stats: dict) -> dict:
    """
    Hitung C_video untuk setiap video.
    Returns dict: video_name → {C_video, category, C_mean, C_std}
    """
    results = {}

    for vname, vs in stats.items():
        n_frames = vs["frames_with_obj"]
        n_list   = vs["n_per_frame"]

        if not n_list or n_frames == 0:
            results[vname] = {
                "C_video": 0.0, "C_mean": 0.0,
                "C_std": 0.0, "category": "low"
            }
            continue

        # Frame-level C(t) hanya untuk frame berisi objek
        C_frames = []
        for n_t in n_list:
            # Estimasi rho_area dan rho_motion dari data objek
            # (tanpa bounding box aktual, gunakan proporsi sebagai proxy)
            rho_area   = min(n_t * 0.02, 1.0)   # estimasi kasar
            n_moving   = vs["obj_warning"] + vs["obj_danger"]
            n_total    = vs["obj_safe"] + n_moving
            rho_motion = n_moving / (n_total + 1e-6)

            C_t = (ALPHA * min(n_t / N_MAX, 1.0)
                   + BETA  * rho_area
                   + GAMMA * rho_motion)
            C_frames.append(C_t)

        C_arr    = np.array(C_frames)
        C_mean   = C_arr.mean()
        C_std    = C_arr.std()
        C_video  = C_mean + LAMBDA_VAR * C_std

        results[vname] = {
            "C_video": round(float(C_video), 4),
            "C_mean":  round(float(C_mean),  4),
            "C_std":   round(float(C_std),   4),
        }

    # Threshold dari distribusi empiris (kuantil 33 dan 67)
    c_vals = [v["C_video"] for v in results.values()]
    t_low  = float(np.percentile(c_vals, 33))
    t_high = float(np.percentile(c_vals, 67))

    for vname in results:
        c = results[vname]["C_video"]
        if c < t_low:
            results[vname]["category"] = "low"
        elif c < t_high:
            results[vname]["category"] = "medium"
        else:
            results[vname]["category"] = "high"

    print(f"  Crowdedness thresholds: low<{t_low:.3f} "
          f"<= medium < {t_high:.3f} <= high")
    return results


# ─────────────────────────────────────────────
# 3. Hitung risk distribution ratio per video
# ─────────────────────────────────────────────

def risk_ratio(stats: dict, video_name: str) -> np.ndarray:
    """
    Kembalikan rasio [safe, warning, danger] pada level frame
    untuk satu video (normalized ke sum=1).
    """
    vs  = stats[video_name]
    arr = np.array([vs["frame_safe"], vs["frame_warning"],
                    vs["frame_danger"]], dtype=float)
    s = arr.sum()
    return arr / s if s > 0 else arr


def split_kl_divergence(
    videos:     list,
    split_assign: dict,
    stats:      dict,
    target_split: str,
) -> float:
    """
    Hitung KL divergence antara distribusi risk di target_split
    dan distribusi risk keseluruhan (sebagai proxy ketidakseimbangan).
    """
    global_arr = np.zeros(3)
    split_arr  = np.zeros(3)

    for v in videos:
        r = risk_ratio(stats, v)
        global_arr += r
        if split_assign.get(v) == target_split:
            split_arr += r

    g = global_arr / (global_arr.sum() + 1e-8)
    s = split_arr  / (split_arr.sum()  + 1e-8)
    s = np.clip(s, 1e-8, 1)

    kl = float(np.sum(g * np.log(g / s + 1e-8)))
    return kl


# ─────────────────────────────────────────────
# 4. Stratified split dengan risk balancing
# ─────────────────────────────────────────────

def stratified_split(
    stats:       dict,
    crowd:       dict,
    train_ratio: float = TRAIN_RATIO,
    val_ratio:   float = VAL_RATIO,
    seed:        int   = 42,
) -> dict:
    """
    Pembagian stratified berdasarkan domain × crowdedness,
    dengan greedy balancing distribusi risk.

    Pendekatan:
      1. Hitung target jumlah video untuk tiap split secara global
      2. Per strata, alokasikan sesuai proporsi strata
      3. Greedy swap untuk menyeimbangkan distribusi risk

    Returns dict: video_name → 'train' | 'val' | 'test'
    """
    rng        = np.random.default_rng(seed)
    assign     = {}
    test_ratio = round(1.0 - train_ratio - val_ratio, 6)

    # Kelompokkan per strata
    strata = defaultdict(list)
    for vname in stats:
        domain   = stats[vname]["domain"]
        category = crowd[vname]["category"]
        strata[(domain, category)].append(vname)

    print(f"\n  Target rasio: train={train_ratio:.2f}  "
          f"val={val_ratio:.2f}  test={test_ratio:.2f}")
    print(f"\n  Strata (domain × crowdedness):")
    for key, vids in sorted(strata.items()):
        print(f"    {key[0]:10s} × {key[1]:6s}  ({len(vids)} video): {vids}")

    for (domain, category), videos in sorted(strata.items()):
        n = len(videos)
        if n == 0:
            continue

        # Urutkan berdasarkan C_video untuk determinisme
        videos_sorted = sorted(videos, key=lambda v: crowd[v]["C_video"])

        if n == 1:
            # Satu video → train (tidak cukup untuk val/test)
            assign[videos_sorted[0]] = "train"

        elif n == 2:
            # Dua video → satu train, satu untuk split minor
            # Pilih split minor berdasarkan rasio yang lebih besar
            minor = "test" if test_ratio >= val_ratio else "val"
            assign[videos_sorted[0]] = minor
            assign[videos_sorted[1]] = "train"

        elif n == 3:
            # Tiga video → test, val, train
            assign[videos_sorted[0]] = "test"
            assign[videos_sorted[1]] = "val"
            assign[videos_sorted[2]] = "train"

        else:
            # n >= 4: alokasi proporsional berdasarkan rasio aktual
            n_test  = max(1, round(n * test_ratio))
            n_val   = max(1, round(n * val_ratio))
            # Pastikan tidak melebihi n
            if n_test + n_val >= n:
                n_test = max(1, n_test - 1) if n_test >= n_val else n_test
                n_val  = max(1, min(n_val, n - n_test - 1))
            n_train = n - n_val - n_test

            # Inisialisasi awal:
            # video dengan risk paling beragam → val/test
            # Ukur keberagaman risk tiap video
            diversity = []
            for v in videos_sorted:
                r = risk_ratio(stats, v)
                # Entropy sebagai ukuran keberagaman
                ent = -float(np.sum(r * np.log(r + 1e-8)))
                diversity.append((v, ent))

            # Sort by diversity descending → video paling beragam ke val/test
            by_diversity = sorted(diversity, key=lambda x: -x[1])

            temp_assign = {v: "train" for v in videos_sorted}
            for idx, (v, _) in enumerate(by_diversity):
                if idx < n_test:
                    temp_assign[v] = "test"
                elif idx < n_test + n_val:
                    temp_assign[v] = "val"

            # Greedy refinement: swap untuk menyeimbangkan risk
            best_assign = dict(temp_assign)
            best_score  = _risk_balance_score(
                videos_sorted, temp_assign, stats
            )

            for i in range(len(videos_sorted)):
                for j in range(i + 1, len(videos_sorted)):
                    v_i = videos_sorted[i]
                    v_j = videos_sorted[j]
                    if temp_assign[v_i] == temp_assign[v_j]:
                        continue
                    trial = dict(temp_assign)
                    trial[v_i], trial[v_j] = trial[v_j], trial[v_i]
                    score = _risk_balance_score(
                        videos_sorted, trial, stats
                    )
                    if score < best_score:
                        best_score  = score
                        best_assign = dict(trial)
                        temp_assign = dict(trial)

            assign.update(best_assign)

    return assign


def _risk_balance_score(videos, assign, stats) -> float:
    """
    Skor keseimbangan risk: jumlah KL divergence val + test
    terhadap distribusi keseluruhan.
    Makin kecil makin seimbang.
    """
    global_arr = np.zeros(3)
    split_arrs = {"train": np.zeros(3), "val": np.zeros(3),
                  "test": np.zeros(3)}

    for v in videos:
        r = risk_ratio(stats, v)
        global_arr           += r
        split_arrs[assign[v]] += r

    score = 0.0
    g = global_arr / (global_arr.sum() + 1e-8)
    g = np.clip(g, 1e-8, 1)

    for split in ["val", "test"]:
        s = split_arrs[split]
        s = s / (s.sum() + 1e-8)
        s = np.clip(s, 1e-8, 1)
        score += float(np.sum(g * np.log(g / s + 1e-8)))

    return score


def fix_missing_domains(assign, stats, crowd) -> dict:
    """
    Post-processing: pastikan setiap domain terwakili di val DAN test.
    Jika ada domain yang tidak ada di val/test, pindahkan satu video
    dari train ke split yang kekurangan domain tersebut.
    Pilih video dengan distribusi risk paling beragam (entropy tertinggi)
    agar val/test tetap informatif.
    """
    assign = dict(assign)
    all_domains = {stats[v]["domain"] for v in stats}

    for split in ["val", "test"]:
        domains_in = {stats[v]["domain"]
                      for v, s in assign.items() if s == split}
        missing = all_domains - domains_in

        for dom in missing:
            # Cari kandidat dari train dengan domain yang hilang
            candidates = [
                v for v, s in assign.items()
                if s == "train" and stats[v]["domain"] == dom
            ]
            if not candidates:
                print(f"  ⚠ Tidak ada video {dom} di train untuk dipindah "
                      f"ke {split}")
                continue

            # Pilih yang paling beragam (entropy tertinggi)
            best_v    = None
            best_ent  = -1
            for v in candidates:
                r   = risk_ratio(stats, v)
                ent = -float(np.sum(r * np.log(r + 1e-8)))
                if ent > best_ent:
                    best_ent = ent
                    best_v   = v

            assign[best_v] = split
            print(f"  ✓ Pindah {best_v} (domain={dom}) dari train ke {split} "
                  f"untuk memenuhi domain coverage")

    return assign


# ─────────────────────────────────────────────
# 5. Validasi dan laporan
# ─────────────────────────────────────────────

def validate_split(assign, stats, crowd) -> bool:
    """
    Pastikan setiap domain terwakili di val dan test.
    """
    ok = True
    for split in ["val", "test"]:
        domains_in_split = {stats[v]["domain"]
                            for v, s in assign.items() if s == split}
        all_domains = {stats[v]["domain"] for v in stats}
        missing = all_domains - domains_in_split
        if missing:
            print(f"  ⚠ {split}: domain tidak terwakili: {missing}")
            ok = False
    return ok


def print_report(assign, stats, crowd):
    """Print laporan distribusi lengkap."""
    splits = ["train", "val", "test"]

    print(f"\n{'='*65}")
    print("HASIL PEMBAGIAN VIDEO")
    print(f"{'='*65}")

    for split in splits:
        videos = sorted(v for v, s in assign.items() if s == split)
        print(f"\n  [{split.upper()}] ({len(videos)} video)")
        for v in videos:
            vs = stats[v]
            c  = crowd[v]
            total_risk = (vs["frame_safe"] + vs["frame_warning"]
                          + vs["frame_danger"])
            if total_risk > 0:
                rs = vs["frame_safe"]    / total_risk * 100
                rw = vs["frame_warning"] / total_risk * 100
                rd = vs["frame_danger"]  / total_risk * 100
            else:
                rs = rw = rd = 0
            print(f"    {v:<20s}  domain={vs['domain']:<9s}  "
                  f"crowd={c['category']:<6s}  "
                  f"risk=[S:{rs:.0f}% W:{rw:.0f}% D:{rd:.0f}%]")

    # Distribusi risk per split
    print(f"\n{'─'*65}")
    print("  DISTRIBUSI RISK PER SPLIT")
    print(f"  {'Split':<8} {'Videos':>6}  "
          f"{'Safe%':>7} {'Warn%':>7} {'Dang%':>7}  "
          f"{'Domain':>20}")
    print(f"  {'─'*63}")

    for split in splits:
        videos = [v for v, s in assign.items() if s == split]
        if not videos:
            continue

        total_s = total_w = total_d = 0
        domain_counts = defaultdict(int)
        for v in videos:
            vs = stats[v]
            total_s += vs["frame_safe"]
            total_w += vs["frame_warning"]
            total_d += vs["frame_danger"]
            domain_counts[vs["domain"]] += 1

        total = total_s + total_w + total_d + 1e-8
        dom_str = " ".join(f"{d}:{n}"
                           for d, n in sorted(domain_counts.items()))
        print(f"  {split:<8} {len(videos):>6}  "
              f"{total_s/total*100:>6.1f}%"
              f"{total_w/total*100:>7.1f}%"
              f"{total_d/total*100:>7.1f}%  "
              f"  {dom_str}")

    # Skor ketidakseimbangan
    print(f"\n  KL divergence dari distribusi global:")
    all_videos = list(stats.keys())
    for split in splits:
        kl = split_kl_divergence(all_videos, assign, stats, split)
        print(f"    {split:<8}: {kl:.4f} {'(ideal=0)' if kl<0.01 else ''}")


def save_report(assign, stats, crowd, out_dir: Path):
    """Simpan hasil ke JSON dan TXT."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON hasil split
    result = {
        "assignment": assign,
        "per_video":  {}
    }
    for v, s in assign.items():
        vs = stats[v]
        c  = crowd[v]
        total_risk = (vs["frame_safe"] + vs["frame_warning"]
                      + vs["frame_danger"]) or 1
        result["per_video"][v] = {
            "split":        s,
            "domain":       vs["domain"],
            "crowdedness":  c["C_video"],
            "crowd_cat":    c["category"],
            "total_frames": vs["total_frames"],
            "frame_safe":   vs["frame_safe"],
            "frame_warning":vs["frame_warning"],
            "frame_danger": vs["frame_danger"],
            "risk_ratio": {
                "safe":    round(vs["frame_safe"]    / total_risk, 4),
                "warning": round(vs["frame_warning"] / total_risk, 4),
                "danger":  round(vs["frame_danger"]  / total_risk, 4),
            },
        }

    json_path = out_dir / "split_result.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Hasil split → {json_path}")

    # TXT untuk laporan ringkas
    txt_path = out_dir / "split_report.txt"
    splits = ["train", "val", "test"]
    with open(txt_path, "w") as f:
        for split in splits:
            videos = sorted(v for v, s in assign.items() if s == split)
            f.write(f"[{split.upper()}]\n")
            for v in videos:
                f.write(f"  {v}\n")
            f.write("\n")
    print(f"  Laporan ringkas → {txt_path}")

    return json_path


# ─────────────────────────────────────────────
# 6. Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Pembagian video train/val/test berdasarkan "
                    "domain, crowdedness, dan distribusi risk"
    )
    parser.add_argument(
        "--ann-dir", required=True,
        help="Folder berisi file anotasi {video}.json "
             "(annotations/full)"
    )
    parser.add_argument(
        "--out", default="split_output",
        help="Folder output hasil (default: split_output)"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.70
    )
    parser.add_argument(
        "--val-ratio",   type=float, default=0.15
    )
    parser.add_argument(
        "--seed", type=int, default=42
    )
    parser.add_argument(
        "--show-only", action="store_true",
        help="Hanya tampilkan laporan, tidak simpan file"
    )
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    out_dir = Path(args.out)

    if not ann_dir.exists():
        raise FileNotFoundError(f"Ann dir tidak ditemukan: {ann_dir}")

    print(f"Ann dir  : {ann_dir}")
    print(f"Out dir  : {out_dir}")
    print(f"Ratio    : train={args.train_ratio} "
          f"val={args.val_ratio} "
          f"test={round(1-args.train_ratio-args.val_ratio,2)}")
    print(f"Seed     : {args.seed}")

    # ── Step 1: Load stats ────────────────────────────────────
    print("\nMemuat anotasi...")
    stats = load_video_stats(ann_dir)
    print(f"  {len(stats)} video ditemukan: {sorted(stats.keys())}")

    # ── Step 2: Crowdedness ───────────────────────────────────
    print("\nMenghitung crowdedness...")
    crowd = compute_crowdedness(stats)
    for v, c in sorted(crowd.items()):
        print(f"  {v:<20s}: C={c['C_video']:.3f}  [{c['category']}]")

    # ── Step 3: Split ─────────────────────────────────────────
    print("\nMenjalankan stratified split...")
    assign = stratified_split(
        stats, crowd,
        train_ratio = args.train_ratio,
        val_ratio   = args.val_ratio,
        seed        = args.seed,
    )

    # ── Step 3b: Fix domain coverage ─────────────────────────
    print("\nMemeriksa domain coverage...")
    assign = fix_missing_domains(assign, stats, crowd)

    # ── Step 4: Validasi ──────────────────────────────────────
    print("\nValidasi...")
    valid = validate_split(assign, stats, crowd)
    if valid:
        print("  ✓ Semua domain terwakili di val dan test")

    # ── Step 5: Laporan ───────────────────────────────────────
    print_report(assign, stats, crowd)

    # ── Step 6: Simpan ────────────────────────────────────────
    if not args.show_only:
        save_report(assign, stats, crowd, out_dir)

    return assign


if __name__ == "__main__":
    main()
