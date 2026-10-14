"""
scripts/evaluate.py
===================
Evaluasi standalone DASTRP-Net pada split tertentu (val/test/train).

Digunakan ketika:
  - Training terputus sebelum evaluasi test selesai
  - Ingin re-evaluate dengan threshold berbeda
  - Ingin evaluasi per-video untuk analisis lebih detail

Kompatibel dengan:
  - architecture.py  : dengan ObjectRiskHead (obj_risk_logits + obj_risk_pred)
  - train.py         : label = max risk per clip (bukan frame terakhir)
  - dataset.py       : DASTRPClipDataset, clip_collate_fn

Output:
  - result_{split}.json   : metrik lengkap frame-level + per-objek
  - result_{split}_per_video.json : breakdown per video
  - Confusion matrix di terminal
"""

import json
import torch
import argparse
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.reproducibility import set_seed, add_seed_args
from models.architecture import DASTRPNet


def build_model(cfg: dict, device):
    """
    Factory function — pilih class model berdasarkan key 'model'
    di config.json. Kompatibel dengan DASTRPNet, ablation, dan
    semua baseline B1–B5 dari train_baseline.py.
    """
    model_name = cfg.get("model", "dastrp").lower()

    if model_name in ("dastrp", ""):
        from models.architecture import DASTRPNet as M
        model = M(
            yolo_weights = cfg["yolo_weights"],
            yolo_conf    = cfg.get("yolo_conf",   0.25),
            yolo_iou     = cfg.get("yolo_iou",    0.45),
            dist_thresh  = cfg.get("dist_thresh", 0.3),
            T_agg        = cfg.get("T_agg",       None),
            ablation     = cfg.get("ablation",    None),
        )
        if cfg.get("ablation"):
            print(f"Ablation mode    : {cfg['ablation']}")

    elif model_name == "b1":
        from models.baseline_b1 import BaselineB1 as M
        model = M(
            yolo_weights = cfg["yolo_weights"],
            yolo_conf    = cfg.get("yolo_conf", 0.25),
            yolo_iou     = cfg.get("yolo_iou",  0.45),
            T_agg        = cfg.get("T_agg",     None),
        )

    elif model_name == "b2":
        from models.baseline_b2 import BaselineB2 as M
        model = M(
            yolo_weights = cfg["yolo_weights"],
            yolo_conf    = cfg.get("yolo_conf",    0.25),
            yolo_iou     = cfg.get("yolo_iou",     0.45),
            lstm_dim     = cfg.get("lstm_dim",     256),
            lstm_layers  = cfg.get("lstm_layers",  2),
            lstm_dropout = cfg.get("lstm_dropout", 0.3),
            T_agg        = cfg.get("T_agg",        None),
        )

    elif model_name == "b3":
        from models.baseline_b3 import BaselineB3 as M
        model = M(
            yolo_weights  = cfg["yolo_weights"],
            yolo_conf     = cfg.get("yolo_conf",    0.25),
            yolo_iou      = cfg.get("yolo_iou",     0.45),
            tf_n_heads    = cfg.get("tf_n_heads",   4),
            tf_n_layers   = cfg.get("tf_n_layers",  2),
            tf_dim_ff     = cfg.get("tf_dim_ff",    512),
            tf_dropout    = cfg.get("tf_dropout",   0.1),
            tf_pool_mode  = cfg.get("tf_pool_mode", "cls"),
            T_agg         = cfg.get("T_agg",        None),
        )

    elif model_name == "b4":
        from models.baseline_b4 import BaselineB4 as M
        model = M(
            yolo_weights = cfg["yolo_weights"],
            yolo_conf    = cfg.get("yolo_conf",      0.25),
            yolo_iou     = cfg.get("yolo_iou",       0.45),
            gcn_hidden   = cfg.get("gcn_hidden",     256),
            gcn_layers   = cfg.get("gcn_layers",     2),
            gcn_dropout  = cfg.get("gcn_dropout",    0.3),
            lstm_dim     = cfg.get("sg_lstm_dim",    256),
            lstm_layers  = cfg.get("sg_lstm_layers", 1),
            lstm_dropout = cfg.get("sg_lstm_dropout",0.2),
            T_agg        = cfg.get("T_agg",          None),
        )

    elif model_name == "b5":
        from models.baseline_b5 import BaselineB5 as M
        model = M(
            base_channels_slow = cfg.get("sf_base_ch",  64),
            alpha              = cfg.get("sf_alpha",     4),
            dropout            = cfg.get("sf_dropout",   0.5),
        )

    else:
        raise ValueError(
            f"Model '{model_name}' tidak dikenal. "
            f"Pilihan: dastrp, b1, b2, b3, b4, b5"
        )

    return model.to(device)
from data.dataset import get_dataloaders, DOMAIN_NAMES, RISK_NAMES


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def per_domain_accuracy(preds, labels, domain_ids) -> dict:
    result = {}
    for d_id, d_name in DOMAIN_NAMES.items():
        mask = (domain_ids == d_id)
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == labels[mask]).float().mean().item() * 100
        result[d_name] = round(acc, 2)
    return result


def per_class_accuracy(preds, labels) -> dict:
    result = {}
    for r_id, r_name in RISK_NAMES.items():
        mask = (labels == r_id)
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == labels[mask]).float().mean().item() * 100
        result[r_name] = round(acc, 2)
    return result


def confusion_matrix_verbose(preds, labels, n: int = 3) -> list:
    cm = torch.zeros(n, n, dtype=torch.long)
    for t, p in zip(labels, preds):
        cm[t.item()][p.item()] += 1
    return cm.tolist()


def print_confusion_matrix(cm: list, split: str):
    names = ["Safe    ", "Warning ", "Danger  "]
    print(f"\n  Confusion Matrix [{split.upper()}]")
    print(f"  {'':12s} {'Pred:Safe':>10} {'Pred:Warn':>10} {'Pred:Danger':>12}")
    print(f"  {'-'*46}")
    for i, row in enumerate(cm):
        total = sum(row)
        acc   = row[i] / total * 100 if total > 0 else 0
        print(f"  True:{names[i]}  {row[0]:>10,}  {row[1]:>10,}  {row[2]:>12,}"
              f"  (recall={acc:.1f}%)")


def compute_f1(preds, labels, num_classes: int = 3) -> dict:
    """Hitung F1 per kelas dan macro-F1."""
    f1s = {}
    for k in range(num_classes):
        tp = ((preds == k) & (labels == k)).sum().float()
        fp = ((preds == k) & (labels != k)).sum().float()
        fn = ((preds != k) & (labels == k)).sum().float()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        name = RISK_NAMES[k]
        f1s[name] = {
            "precision": round(prec.item() * 100, 2),
            "recall":    round(rec.item()  * 100, 2),
            "f1":        round(f1.item()   * 100, 2),
        }
    macro_f1 = sum(v["f1"] for v in f1s.values()) / num_classes
    f1s["macro"] = {"f1": round(macro_f1, 2)}
    return f1s


# ─────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:      DASTRPNet,
    loader:     torch.utils.data.DataLoader,
    device:     torch.device,
    split:      str  = "test",
    use_amp:    bool = True,
) -> dict:
    """
    Evaluasi lengkap: frame-level + per-objek risk.
    Label frame-level = max risk dalam clip (konsisten dengan train.py).
    """
    model.eval()

    # Frame-level
    all_preds    = []
    all_labels   = []
    all_domains  = []
    all_uncs     = []
    all_probs    = []

    # Per-objek risk
    all_obj_preds  = []
    all_obj_labels = []

    # Per-video tracking
    video_stats = defaultdict(lambda: {
        "n_clips": 0,
        "correct": 0,
        "labels":  [],
        "preds":   [],
    })

    pbar = tqdm(loader, desc=f"  Evaluating [{split}]",
                unit="batch", dynamic_ncols=True, leave=True)

    for batch in pbar:
        clip_images = batch["images"].to(device, non_blocking=True)
        domain_ids  = batch["domain_ids"].to(device, non_blocking=True)
        T           = batch["frame_risks"].size(1)

        with torch.amp.autocast("cuda",
                                enabled=(use_amp and device.type == "cuda")):
            outputs = model(
                clip_images  = clip_images,
                domain_ids   = None,    # inference: pakai scene classifier
                gt_boxes_seq = None,    # inference: pakai YOLO detection
            )

        # ── Frame-level ───────────────────────────────────────
        preds  = outputs["risk_pred"].cpu()
        labels = batch["frame_risks"].max(dim=1).values   # max risk per clip
        doms   = domain_ids.cpu()
        uncs   = outputs["uncertainty"].cpu()

        all_preds.append(preds)
        all_labels.append(labels)
        all_domains.append(doms)
        all_uncs.append(uncs)
        all_probs.append(outputs["risk_prob"].cpu())

        # Per-video tracking
        for b, vname in enumerate(batch["video_names"]):
            video_stats[vname]["n_clips"] += 1
            video_stats[vname]["correct"] += (preds[b] == labels[b]).item()
            video_stats[vname]["labels"].append(labels[b].item())
            video_stats[vname]["preds"].append(preds[b].item())

        # ── Per-objek risk ────────────────────────────────────
        B = clip_images.size(0)
        for b in range(B):
            obj_pred = outputs["obj_risk_pred"][b].cpu()   # (N_i,)
            obj_lbl  = batch["risk_seq"][b][T - 1]         # (N_i,)
            if obj_pred.size(0) > 0 and obj_lbl.size(0) > 0:
                n = min(obj_pred.size(0), obj_lbl.size(0))
                all_obj_preds.append(obj_pred[:n])
                all_obj_labels.append(obj_lbl[:n])

    # ── Aggregate frame-level ─────────────────────────────────
    preds   = torch.cat(all_preds)
    labels  = torch.cat(all_labels)
    domains = torch.cat(all_domains)
    uncs    = torch.cat(all_uncs)
    probs   = torch.cat(all_probs)   # (N, 3)

    # ── Calibration metrics ──────────────────────────────
    p_correct = probs[torch.arange(len(labels)), labels].clamp(min=1e-10)
    nll       = -p_correct.log().mean().item()
    K         = probs.size(1)
    onehot    = torch.zeros_like(probs)
    onehot.scatter_(1, labels.unsqueeze(1), 1.0)
    brier     = ((probs - onehot) ** 2).sum(dim=1).mean().item()
    nll_random   = float(torch.tensor(float(K)).log())
    brier_random = 2.0 * (1.0 - 1.0 / K)

    overall  = (preds == labels).float().mean().item() * 100
    dom_acc  = per_domain_accuracy(preds, labels, domains)
    cls_acc  = per_class_accuracy(preds, labels)
    cm       = confusion_matrix_verbose(preds, labels)
    f1       = compute_f1(preds, labels)

    pred_dist = {r_name: (preds == r_id).sum().item()
                 for r_id, r_name in RISK_NAMES.items()}
    true_dist = {r_name: (labels == r_id).sum().item()
                 for r_id, r_name in RISK_NAMES.items()}

    unc_dom = {}
    for d_id, d_name in DOMAIN_NAMES.items():
        mask = (domains == d_id)
        if mask.sum() > 0:
            unc_dom[d_name] = round(uncs[mask].mean().item(), 4)

    # ── Aggregate per-objek ───────────────────────────────────
    obj_overall  = 0.0
    obj_cls_acc  = {}
    obj_f1       = {}
    obj_cm       = []
    obj_pred_dist = {}

    if all_obj_preds:
        op = torch.cat(all_obj_preds)
        ol = torch.cat(all_obj_labels)
        obj_overall   = (op == ol).float().mean().item() * 100
        obj_cls_acc   = per_class_accuracy(op, ol)
        obj_f1        = compute_f1(op, ol)
        obj_cm        = confusion_matrix_verbose(op, ol)
        obj_pred_dist = {r_name: (op == r_id).sum().item()
                         for r_id, r_name in RISK_NAMES.items()}

    # ── Per-video summary ─────────────────────────────────────
    per_video = {}
    for vname, vs in sorted(video_stats.items()):
        vp = torch.tensor(vs["preds"])
        vl = torch.tensor(vs["labels"])
        per_video[vname] = {
            "n_clips": vs["n_clips"],
            "acc":     round((vp == vl).float().mean().item() * 100, 2),
            "class_acc": per_class_accuracy(vp, vl),
            "pred_dist": {r_name: (vp == r_id).sum().item()
                          for r_id, r_name in RISK_NAMES.items()},
        }

    # ── Print summary ─────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"EVALUATION RESULTS — {split.upper()}")
    print(f"{'='*65}")
    print(f"  Overall acc    : {overall:.2f}%")
    print(f"  Per-domain     : {dom_acc}")
    print(f"  Per-class      : {cls_acc}")
    print(f"  Macro F1       : {f1['macro']['f1']:.2f}%")
    print(f"\n  Per-class F1:")
    for r_name, metrics in f1.items():
        if r_name == "macro":
            continue
        print(f"    {r_name:<8}: "
              f"P={metrics['precision']:.1f}%  "
              f"R={metrics['recall']:.1f}%  "
              f"F1={metrics['f1']:.1f}%")
    print(f"\n  Uncertainty    : {unc_dom}")
    print(f"  NLL            : {nll:.4f}  (random: {nll_random:.4f})")
    print(f"  Brier Score    : {brier:.4f}  (random: {brier_random:.4f})")
    print(f"  True dist      : {true_dist}")
    print(f"  Pred dist      : {pred_dist}")
    print_confusion_matrix(cm, split)

    print(f"\n  [Per-objek risk]")
    print(f"  Overall acc    : {obj_overall:.2f}%")
    print(f"  Per-class      : {obj_cls_acc}")
    if obj_f1:
        print(f"  Macro F1       : {obj_f1['macro']['f1']:.2f}%")
    print(f"  Pred dist      : {obj_pred_dist}")
    if obj_cm:
        print_confusion_matrix(obj_cm, f"{split}_obj")

    print(f"\n  Per-video:")
    for vname, vs in per_video.items():
        print(f"    {vname:<20}: acc={vs['acc']:.1f}%  "
              f"pred={vs['pred_dist']}")

    return {
        # Frame-level
        f"{split}_acc":           round(overall, 2),
        f"{split}_domain_acc":    dom_acc,
        f"{split}_class_acc":     cls_acc,
        f"{split}_f1":            f1,
        f"{split}_confusion":     cm,
        f"{split}_unc":           unc_dom,
        f"{split}_nll":           round(nll, 4),
        f"{split}_brier":         round(brier, 4),
        f"{split}_nll_random":    round(nll_random, 4),
        f"{split}_brier_random":  round(brier_random, 4),
        f"{split}_pred_dist":     pred_dist,
        f"{split}_true_dist":     true_dist,
        # Per-objek
        f"{split}_obj_risk_acc":  round(obj_overall, 2),
        f"{split}_obj_class_acc": obj_cls_acc,
        f"{split}_obj_f1":        obj_f1,
        f"{split}_obj_cm":        obj_cm,
        f"{split}_obj_pred_dist": obj_pred_dist,
        # Uncertainty per-clip (untuk distribusi plot fig:uncertainty)
        f"{split}_unc_detail": {
            "uncertainty": uncs.tolist(),
            "label":       labels.tolist(),
            "domain":      domains.tolist(),
            "pred":        preds.tolist(),
            "risk_prob":   probs.tolist(),
        },
        # Per-video
        f"{split}_per_video":     per_video,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run_test(
    config_path:  str,
    split:        str = "test",
    checkpoint:   str = None,
    dataset_root: str = None,
    yolo_weights: str = None,
    out_dir:      str = None,
    num_workers:  int = None,
    batch_size:   int = None,
):
    """
    Evaluate one trained run.

    By default everything comes from config.json (written by train.py):
    the checkpoint is <save_dir>/best_model.pth and results are written
    next to it. Every path can be overridden, which is how the released
    weights are evaluated, e.g.
        run_test("weights/dastrp_exp3_config.json", "test",
                 checkpoint="weights/dastrp_exp3.pth",
                 dataset_root="data/processed/dastrp",
                 yolo_weights="weights/yolo11n_motor.pt",
                 out_dir="runs/eval_release")
    """
    # ── Load config ───────────────────────────────────────────
    with open(config_path) as f:
        cfg = json.load(f)
    if dataset_root:
        cfg["dataset_root"] = dataset_root
    if yolo_weights:
        cfg["yolo_weights"] = yolo_weights
    if num_workers is not None:
        cfg["num_workers"] = num_workers
    if batch_size is not None:
        cfg["batch_size"] = batch_size

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = (Path(checkpoint) if checkpoint
                 else Path(cfg["save_dir"]) / "best_model.pth")
    save_dir  = (Path(out_dir) if out_dir
                 else Path(cfg["save_dir"]) if not checkpoint
                 else ckpt_path.parent)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device     : {device}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Dataset    : {cfg['dataset_root']}")
    print(f"Split      : {split}")
    print()

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan: {ckpt_path}\n"
            f"Pastikan training sudah selesai atau gunakan --checkpoint "
            f"untuk path lain."
        )

    # ── Load model ────────────────────────────────────────────
    ckpt       = torch.load(ckpt_path, map_location=device)
    model_name = cfg.get("model", "dastrp")
    print(f"Model type       : {model_name}")
    model      = build_model(cfg, device)

    state_dict = ckpt["model_state"]
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        # Handle torch.compile prefix
        new_sd = {k.replace("_orig_mod.", ""): v
                  for k, v in state_dict.items()}
        model.load_state_dict(new_sd)

    print(f"Model loaded. "
          f"Best val_acc: {ckpt.get('val_acc', 'N/A')}%  "
          f"(epoch {ckpt.get('epoch', '?')})")

    # ── DataLoader ────────────────────────────────────────────
    # Hanya build split yang diminta — lebih cepat dan hemat memori
    # dibanding membangun train+val+test sekaligus
    loaders = get_dataloaders(
        dataset_root = cfg["dataset_root"],
        T            = cfg.get("T",           4),
        stride       = cfg.get("T",           4),   # non-overlapping untuk eval
        img_size     = cfg.get("img_size",    640),
        batch_size   = cfg.get("batch_size",  16),
        num_workers  = cfg.get("num_workers", 4),
        seed         = cfg.get("seed",        42),
        splits       = [split],
    )

    if split not in loaders:
        raise ValueError(
            f"Split '{split}' tidak ditemukan di dataset. "
            f"Tersedia: {list(loaders.keys())}"
        )

    # ── Evaluate ──────────────────────────────────────────────
    use_amp = cfg.get("use_amp", True) and device.type == "cuda"
    stats   = evaluate(model, loaders[split], device, split, use_amp)

    # ── Simpan hasil ──────────────────────────────────────────
    # File utama
    out_path = save_dir / f"result_{split}.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nHasil disimpan → {out_path}")

    # File per-video terpisah (lebih mudah dibaca)
    per_video_key = f"{split}_per_video"
    if per_video_key in stats:
        pv_path = save_dir / f"result_{split}_per_video.json"
        with open(pv_path, "w") as f:
            json.dump(stats[per_video_key], f, indent=2)
        print(f"Per-video   → {pv_path}")

    # ── Update history.json jika ada ─────────────────────────
    history_path = save_dir / "history.json"
    if history_path.exists() and not checkpoint:
        with open(history_path) as f:
            history = json.load(f)

        # Cek apakah sudah ada final_test
        key = f"final_{split}"
        has_key = any(key in h for h in history)

        if not has_key:
            history.append({key: stats})
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)
            print(f"history.json → ditambahkan entri '{key}'")
        else:
            print(f"history.json → entri '{key}' sudah ada, tidak ditimpa")

    return stats


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone evaluation of DASTRP-Net or a baseline "
                    "(training run or released weights)."
    )
    parser.add_argument("--config", required=True,
                        help="config.json of a training run, or "
                             "weights/<name>_config.json for released weights")
    parser.add_argument("--split", default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint file (default: <save_dir>/best_model.pth)")
    parser.add_argument("--dataset", default=None,
                        help="Override dataset_root from the config")
    parser.add_argument("--yolo", default=None,
                        help="Override yolo_weights from the config")
    parser.add_argument("--out-dir", default=None,
                        help="Where to write result_<split>.json "
                             "(default: run dir, or the checkpoint's folder)")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    add_seed_args(parser)
    args = parser.parse_args()
    set_seed(args.seed, args.deterministic)

    run_test(args.config, args.split,
             checkpoint   = args.checkpoint,
             dataset_root = args.dataset,
             yolo_weights = args.yolo,
             out_dir      = args.out_dir,
             num_workers  = args.workers,
             batch_size   = args.batch)
