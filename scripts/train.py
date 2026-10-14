"""
scripts/train.py
================
Training loop DASTRP-Net berbasis clip (sequence-aware batching).
Dioptimalkan untuk NVIDIA RTX 5000 Ada Generation (30 GB VRAM, CUDA 12.4).

Optimasi GPU:
  - Automatic Mixed Precision (AMP)
  - cudnn.benchmark untuk input size tetap
  - num_workers=8, prefetch_factor=2
  - VRAM monitoring per epoch
  - TensorBoard logging

Loss multi-task:
  L_total = L_edl_agg     (EDL pada agregasi T frame)
          + λ_frame      · L_frame      (focal loss per frame)
          + λ_domain     · L_domain     (auxiliary scene classifier)
          + λ_motion     · L_motion     (gerak/diam per objek)
          + λ_obj_risk   · L_obj_risk   (per-objek risk level)
          + λ_unc        · L_unc_reg    (uncertainty reg Ringroad)
"""

import math
import json
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.architecture import DASTRPNet, EvidentialRiskHead
from utils.reproducibility import set_seed, add_seed_args
from data.dataset import get_dataloaders, DOMAIN_NAMES, RISK_NAMES


# ─────────────────────────────────────────────
# Default config
# ─────────────────────────────────────────────

DEFAULT_CFG = {
    # Data
    "dataset_root":    "dataset",
    "img_size":        640,
    "T":               4,
    "stride":          2,
    "batch_size":      16,
    "num_workers":     8,
    "prefetch_factor": 2,

    # Model
    "yolo_weights":    "path/to/yolo11n/best.pt",
    "yolo_conf":       0.25,
    "yolo_iou":        0.45,
    "dist_thresh":     0.3,
    "T_agg":           None,

    # Training
    "epochs":          100,
    "lr":              1e-4,
    "weight_decay":    1e-4,
    "warmup_epochs":   2,
    "grad_clip":       5.0,
    "patience":        20,

    # Ablation study
    "ablation":        None,   # None | no_gnn | no_cbn | no_fusion |
                                # no_edl | no_velocity | no_obj_risk

    # GPU
    "use_amp":         True,
    "use_compile":     False,
    "compile_mode":    "reduce-overhead",
    "use_tensorboard": True,

    # Loss weights
    "lambda_frame":    1.0,
    "lambda_domain":   0.2,
    "lambda_motion":   0.2,
    "lambda_obj_risk": 0.3,
    "lambda_unc":      0.1,
    "unc_target":      0.4,

    # Teacher forcing
    "use_gt_boxes":    True,

    # Checkpoint
    "save_dir":        "runs/exp1",
    "save_every":      10,
    "resume":          None,

    # Reproducibility
    "seed":            42,
    "deterministic":   False,
}


# ─────────────────────────────────────────────
# Metric tracker
# ─────────────────────────────────────────────

class MetricTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.data = defaultdict(list)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            val = v.item() if isinstance(v, torch.Tensor) else float(v)
            self.data[k].append(val)

    def mean(self, key) -> float:
        vals = self.data[key]
        return sum(vals) / len(vals) if vals else 0.0

    def summary(self) -> dict:
        return {k: round(self.mean(k), 6) for k in self.data}


# ─────────────────────────────────────────────
# VRAM monitor
# ─────────────────────────────────────────────

def log_vram(label: str = "") -> dict:
    if not torch.cuda.is_available():
        return {}
    reserved = torch.cuda.memory_reserved()  / 1e9
    peak     = torch.cuda.max_memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()
    if label:
        print(f"  VRAM [{label}]: {reserved:.2f} GB reserved | "
              f"{peak:.2f} GB peak")
    return {"vram_reserved_gb": round(reserved, 3),
            "vram_peak_gb":     round(peak, 3)}


# ─────────────────────────────────────────────
# Class weight computation
# ─────────────────────────────────────────────

def compute_class_weights(loaders, device) -> torch.Tensor:
    """
    Hitung inverse-frequency weight dari distribusi label clip aktual.
    Label clip = max risk dalam clip (bukan frame terakhir).
    Lebih akurat dari meta.json karena menghitung di level clip.
    """
    counts = torch.zeros(3)
    print("Menghitung distribusi label clip training...")
    for batch in loaders["train"]:
        labels = batch["frame_risks"].max(dim=1).values  # (B,)
        for k in range(3):
            counts[k] += (labels == k).sum().item()

    total   = counts.sum()
    weights = total / (3 * counts.clamp(min=1))
    print(f"  Clip counts  : Safe={int(counts[0])}, "
          f"Warning={int(counts[1])}, Danger={int(counts[2])}")
    print(f"  Class weights: Safe={weights[0]:.3f}, "
          f"Warning={weights[1]:.3f}, Danger={weights[2]:.3f}")
    return weights.to(device)


# ─────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────

def focal_loss(
    logits:        torch.Tensor,
    labels:        torch.Tensor,
    class_weights: torch.Tensor = None,
    gamma:         float        = 2.0,
) -> torch.Tensor:
    """
    Focal loss (Lin et al. 2017): (1 - p_t)^gamma * CE
    Fokus pada hard examples (Warning/Danger yang susah diprediksi).
    """
    ce = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def compute_loss(
    outputs:       dict,
    batch:         dict,
    epoch:         int,
    num_epochs:    int,
    cfg:           dict,
    device:        torch.device,
    class_weights: torch.Tensor = None,
) -> dict:
    """
    Multi-task loss.

    Komponen:
      1. L_edl_agg   — EDL pada clip-level embedding (label = max risk clip)
      2. L_frame     — focal loss per frame dalam clip
      3. L_domain    — auxiliary CE untuk scene classifier
      4. L_motion    — CE gerak/diam per objek, frame terakhir
      5. L_obj_risk  — CE per-objek risk, frame terakhir
      6. L_unc       — uncertainty regularization untuk Ringroad
    """
    B          = outputs["alpha"].size(0)
    T          = batch["frame_risks"].size(1)
    domain_ids = batch["domain_ids"].to(device)

    # ── 1. EDL loss clip-level ────────────────────────────────
    # Label = risk tertinggi dalam clip (bukan frame terakhir)
    agg_labels = batch["frame_risks"].to(device).max(dim=1).values
    if cfg.get("ablation") == "no_edl":
        # A4: standard (class-weighted) cross-entropy instead of EDL
        l_edl_agg = F.cross_entropy(outputs["risk_logits"], agg_labels,
                                    weight=class_weights)
    else:
        l_edl_agg = EvidentialRiskHead.edl_loss(
            outputs["alpha"], agg_labels, epoch, num_epochs
        )

    # ── 2. Focal loss per frame ───────────────────────────────
    frame_risk_labels = batch["frame_risks"].to(device)   # (B, T)
    all_frame_probs   = outputs["all_frame_risks"]        # (B, T, 3)

    per_frame_losses = []
    for t in range(T):
        logits_t = all_frame_probs[:, t].clamp(min=1e-8).log()   # (B, 3)
        per_frame_losses.append(
            focal_loss(logits_t, frame_risk_labels[:, t], class_weights)
        )
    l_frame = torch.stack(per_frame_losses).mean()

    # ── 3. Domain auxiliary loss ──────────────────────────────
    l_domain = nn.CrossEntropyLoss()(outputs["domain_logits"], domain_ids)

    # ── 4. Motion loss — gerak/diam per objek ─────────────────
    motion_losses = []
    for b in range(B):
        mot_logits = outputs["motion_logits"][b]               # (N_i, 2)
        labels_b   = batch["labels_seq"][b][T - 1].to(device) # (N_i,)
        if mot_logits.size(0) == 0 or labels_b.size(0) == 0:
            continue
        n = min(mot_logits.size(0), labels_b.size(0))
        motion_losses.append(
            nn.CrossEntropyLoss()(mot_logits[:n], labels_b[:n])
        )
    l_motion = (torch.stack(motion_losses).mean()
                if motion_losses
                else torch.tensor(0.0, device=device))

    # ── 5. Per-objek risk loss ────────────────────────────────
    # Supervised oleh per-objek risk annotation dari tracks JSON
    obj_risk_losses = []
    for b in (range(B) if cfg.get("ablation") != "no_obj_risk" else []):
        obj_logits = outputs["obj_risk_logits"][b]              # (N_i, 3)
        obj_labels = batch["risk_seq"][b][T - 1].to(device)    # (N_i,)
        if obj_logits.size(0) == 0 or obj_labels.size(0) == 0:
            continue
        n = min(obj_logits.size(0), obj_labels.size(0))
        obj_risk_losses.append(
            nn.CrossEntropyLoss(weight=class_weights)(
                obj_logits[:n], obj_labels[:n]
            )
        )
    l_obj_risk = (torch.stack(obj_risk_losses).mean()
                  if obj_risk_losses
                  else torch.tensor(0.0, device=device))

    # ── 6. Uncertainty reg — Ringroad ─────────────────────────
    unc       = outputs["uncertainty"]
    ring_mask = (domain_ids == 2)
    l_unc = (torch.relu(cfg["unc_target"] - unc[ring_mask]).mean()
             if ring_mask.sum() > 0
             else torch.tensor(0.0, device=device))

    # ── Total ─────────────────────────────────────────────────
    l_total = (l_edl_agg
               + cfg["lambda_frame"]    * l_frame
               + cfg["lambda_domain"]   * l_domain
               + cfg["lambda_motion"]   * l_motion
               + cfg["lambda_obj_risk"] * l_obj_risk
               + cfg["lambda_unc"]      * l_unc)

    return {
        "total":    l_total,
        "edl_agg":  l_edl_agg.detach(),
        "frame":    l_frame.detach(),
        "domain":   l_domain.detach(),
        "motion":   l_motion.detach(),
        "obj_risk": l_obj_risk.detach(),
        "unc":      l_unc.detach(),
    }


# ─────────────────────────────────────────────
# Evaluation helpers
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


def confusion_matrix(preds, labels, n: int = 3) -> list:
    cm = torch.zeros(n, n, dtype=torch.long)
    for t, p in zip(labels, preds):
        cm[t.item()][p.item()] += 1
    return cm.tolist()


def per_class_accuracy_obj(all_obj_preds, all_obj_labels) -> dict:
    """
    Hitung per-class accuracy untuk per-objek risk prediction.
    all_obj_preds / all_obj_labels: list of tensor (N_i,) yang digabung.
    """
    if not all_obj_preds:
        return {}
    preds  = torch.cat(all_obj_preds)
    labels = torch.cat(all_obj_labels)
    return per_class_accuracy(preds, labels)


# ─────────────────────────────────────────────
# LR scheduler
# ─────────────────────────────────────────────

def make_scheduler(optimizer, cfg, n_steps_per_epoch):
    total_steps  = cfg["epochs"] * n_steps_per_epoch
    warmup_steps = cfg["warmup_epochs"] * n_steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────
# Train one epoch
# ─────────────────────────────────────────────

def train_epoch(
    model:         nn.Module,
    loader:        torch.utils.data.DataLoader,
    optimizer:     torch.optim.Optimizer,
    scheduler:     torch.optim.lr_scheduler.LambdaLR,
    scaler:        torch.amp.GradScaler,
    epoch:         int,
    num_epochs:    int,
    device:        torch.device,
    cfg:           dict,
    class_weights: torch.Tensor = None,
) -> dict:
    model.train()
    model.yolo.backbone.eval()   # YOLO selalu eval

    tracker = MetricTracker()
    t0      = time.time()
    use_amp = cfg["use_amp"] and device.type == "cuda"

    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{num_epochs}",
                unit="batch", dynamic_ncols=True, leave=False)

    for step, batch in enumerate(pbar):
        clip_images  = batch["images"].to(device, non_blocking=True)
        domain_ids   = batch["domain_ids"].to(device, non_blocking=True)
        gt_boxes_seq = batch["boxes_seq"] if cfg["use_gt_boxes"] else None

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(
                clip_images  = clip_images,
                domain_ids   = domain_ids,
                gt_boxes_seq = gt_boxes_seq,
                epoch        = epoch,
                num_epochs   = num_epochs,
            )
            losses = compute_loss(
                outputs, batch, epoch, num_epochs, cfg, device, class_weights
            )

        scaler.scale(losses["total"]).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=cfg["grad_clip"],
        )
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # Accuracy frame-level (label = max risk clip)
        pred_risk = outputs["risk_pred"]
        true_risk = batch["frame_risks"].max(dim=1).values.to(device)
        acc       = (pred_risk == true_risk).float().mean()

        tracker.update(
            loss          = losses["total"].detach(),
            loss_edl      = losses["edl_agg"],
            loss_frame    = losses["frame"],
            loss_domain   = losses["domain"],
            loss_motion   = losses["motion"],
            loss_obj_risk = losses["obj_risk"],
            acc           = acc,
        )

        pbar.set_postfix({
            "loss":     f"{tracker.mean('loss'):.4f}",
            "edl":      f"{tracker.mean('loss_edl'):.4f}",
            "obj_risk": f"{tracker.mean('loss_obj_risk'):.4f}",
            "acc":      f"{tracker.mean('acc')*100:.1f}%",
            "lr":       f"{scheduler.get_last_lr()[0]:.1e}",
        }, refresh=False)

    summary         = tracker.summary()
    summary["time"] = round(time.time() - t0, 1)
    return summary


# ─────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model:   nn.Module,
    loader:  torch.utils.data.DataLoader,
    device:  torch.device,
    split:   str  = "val",
    use_amp: bool = True,
) -> dict:
    """
    Evaluasi inference mode (domain + box dari model sendiri).
    Label frame-level: max risk dalam clip.
    Label per-objek: dari tracks annotation.
    """
    model.eval()

    all_preds   = []
    all_labels  = []
    all_domains = []
    all_uncs    = []

    # Per-objek risk evaluation
    all_obj_preds  = []
    all_obj_labels = []

    pbar = tqdm(loader, desc=f"  Eval [{split}]",
                unit="batch", dynamic_ncols=True, leave=False)

    for batch in pbar:
        clip_images = batch["images"].to(device, non_blocking=True)
        domain_ids  = batch["domain_ids"].to(device, non_blocking=True)
        T           = batch["frame_risks"].size(1)

        with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
            outputs = model(
                clip_images  = clip_images,
                domain_ids   = None,
                gt_boxes_seq = None,
            )

        # Frame-level
        all_preds.append(outputs["risk_pred"].cpu())
        all_labels.append(batch["frame_risks"].max(dim=1).values)
        all_domains.append(domain_ids.cpu())
        all_uncs.append(outputs["uncertainty"].cpu())

        # Per-objek risk — kumpulkan dari frame terakhir
        B = clip_images.size(0)
        for b in range(B):
            obj_pred = outputs["obj_risk_pred"][b].cpu()   # (N_i,)
            obj_lbl  = batch["risk_seq"][b][T - 1]         # (N_i,)
            if obj_pred.size(0) > 0 and obj_lbl.size(0) > 0:
                n = min(obj_pred.size(0), obj_lbl.size(0))
                all_obj_preds.append(obj_pred[:n])
                all_obj_labels.append(obj_lbl[:n])

    # ── Frame-level metrics ───────────────────────────────────
    preds   = torch.cat(all_preds)
    labels  = torch.cat(all_labels)
    domains = torch.cat(all_domains)
    uncs    = torch.cat(all_uncs)

    overall  = (preds == labels).float().mean().item() * 100
    dom_acc  = per_domain_accuracy(preds, labels, domains)
    cls_acc  = per_class_accuracy(preds, labels)
    cm       = confusion_matrix(preds, labels)
    pred_dist = {r_name: (preds == r_id).sum().item()
                 for r_id, r_name in RISK_NAMES.items()}

    unc_dom = {}
    for d_id, d_name in DOMAIN_NAMES.items():
        mask = (domains == d_id)
        if mask.sum() > 0:
            unc_dom[d_name] = round(uncs[mask].mean().item(), 4)

    # ── Per-objek risk metrics ────────────────────────────────
    obj_cls_acc = per_class_accuracy_obj(all_obj_preds, all_obj_labels)
    obj_overall = 0.0
    if all_obj_preds:
        op = torch.cat(all_obj_preds)
        ol = torch.cat(all_obj_labels)
        obj_overall = (op == ol).float().mean().item() * 100

    print(f"\n  [{split.upper()}]  acc={overall:.2f}%  "
          f"| per-domain={dom_acc}")
    print(f"           per-class={cls_acc}")
    print(f"           obj_risk_acc={obj_overall:.2f}%  "
          f"obj_per-class={obj_cls_acc}")
    print(f"           unc={unc_dom}  | pred-dist={pred_dist}")

    return {
        f"{split}_acc":            round(overall, 2),
        f"{split}_domain_acc":     dom_acc,
        f"{split}_class_acc":      cls_acc,
        f"{split}_confusion":      cm,
        f"{split}_unc":            unc_dom,
        f"{split}_pred_dist":      pred_dist,
        f"{split}_obj_risk_acc":   round(obj_overall, 2),
        f"{split}_obj_class_acc":  obj_cls_acc,
    }


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────

def train(cfg: dict):
    # ── Setup ─────────────────────────────────────────────────
    set_seed(cfg.get("seed", 42), cfg.get("deterministic", False))
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if not cfg.get("deterministic", False):
            torch.backends.cudnn.benchmark     = True
            torch.backends.cudnn.deterministic = False
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}  ({gpu_mem:.1f} GB)")

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    if cfg.get("use_tensorboard", True):
        writer = SummaryWriter(log_dir=str(save_dir / "tensorboard"))
        print(f"TensorBoard : tensorboard --logdir {save_dir / 'tensorboard'}")

    print("=" * 65)
    print("DASTRP-Net Training")
    print("=" * 65)
    print(f"Device    : {device}")
    print(f"AMP       : {'on' if cfg['use_amp'] else 'off'}")
    print(f"YOLO      : {cfg['yolo_weights']}")
    print(f"Dataset   : {cfg['dataset_root']}")
    print(f"Clip T={cfg['T']}, stride={cfg['stride']}, "
          f"batch={cfg['batch_size']}")
    print(f"Epochs    : {cfg['epochs']} | patience={cfg.get('patience',20)}")
    print()

    with open(save_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ── DataLoaders ───────────────────────────────────────────
    loaders = get_dataloaders(
        dataset_root   = cfg["dataset_root"],
        T              = cfg["T"],
        stride         = cfg["stride"],
        img_size       = cfg["img_size"],
        batch_size     = cfg["batch_size"],
        num_workers    = cfg["num_workers"],
        seed           = cfg.get("seed", 42),
    )
    assert "train" in loaders, "Training split tidak ditemukan!"

    # ── Class weights ─────────────────────────────────────────
    class_weights = compute_class_weights(loaders, device)

    # ── Model ─────────────────────────────────────────────────
    model = DASTRPNet(
        yolo_weights = cfg["yolo_weights"],
        yolo_conf    = cfg["yolo_conf"],
        yolo_iou     = cfg["yolo_iou"],
        dist_thresh  = cfg["dist_thresh"],
        T_agg        = cfg["T_agg"],
        ablation     = cfg.get("ablation", None),
    ).to(device)

    if cfg.get("ablation"):
        print(f"Ablation mode    : {cfg['ablation']}")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_yolo  = sum(p.numel() for p in model.yolo.parameters())
    print(f"Trainable params : {n_train:,}")
    print(f"Frozen (YOLO)    : {n_yolo:,}")

    if cfg["use_compile"] and hasattr(torch, "compile"):
        print(f"Compiling model...")
        model = torch.compile(model, mode=cfg["compile_mode"])
        print("Compile selesai.")
    print()

    # ── AMP Scaler ────────────────────────────────────────────
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(cfg["use_amp"] and device.type == "cuda")
    )

    # ── Optimizer ─────────────────────────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
        fused        = (device.type == "cuda"),
    )
    scheduler = make_scheduler(optimizer, cfg, len(loaders["train"]))

    # ── Resume ────────────────────────────────────────────────
    start_epoch = 1
    best_val    = 0.0
    best_ep     = 0

    if cfg.get("resume") and Path(cfg["resume"]).exists():
        print(f"Resuming dari: {cfg['resume']}")
        ckpt       = torch.load(cfg["resume"], map_location=device)
        state_dict = ckpt["model_state"]
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            new_sd = {k.replace("_orig_mod.", ""): v
                      for k, v in state_dict.items()}
            model.load_state_dict(new_sd)
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val    = ckpt.get("val_acc", 0.0)
        best_ep     = ckpt.get("epoch", 0)
        print(f"  Lanjut dari epoch {start_epoch}, "
              f"best_val={best_val:.2f}%")
        print()

    log_vram("sebelum training")

    # ── Training loop ─────────────────────────────────────────
    history          = []
    use_amp          = cfg["use_amp"] and device.type == "cuda"
    epochs_no_impr   = 0
    patience         = cfg.get("patience", 20)

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        lr_now = scheduler.get_last_lr()[0]
        print(f"\n{'='*65}")
        print(f"Epoch {epoch}/{cfg['epochs']}  |  LR={lr_now:.2e}  |  "
              f"AMP={'on' if use_amp else 'off'}")
        print("=" * 65)

        tr = train_epoch(
            model, loaders["train"], optimizer, scheduler, scaler,
            epoch, cfg["epochs"], device, cfg, class_weights,
        )
        vram_tr = log_vram("train")

        print(f"\n  [TRAIN]  "
              f"loss={tr['loss']:.4f}  "
              f"edl={tr['loss_edl']:.4f}  "
              f"obj_risk={tr['loss_obj_risk']:.4f}  "
              f"acc={tr['acc']*100:.2f}%  "
              f"t={tr['time']}s")

        # ── TensorBoard train ─────────────────────────────────
        if writer:
            writer.add_scalar("train/loss_total",    tr["loss"],          epoch)
            writer.add_scalar("train/loss_edl",      tr["loss_edl"],      epoch)
            writer.add_scalar("train/loss_frame",    tr["loss_frame"],    epoch)
            writer.add_scalar("train/loss_domain",   tr["loss_domain"],   epoch)
            writer.add_scalar("train/loss_motion",   tr["loss_motion"],   epoch)
            writer.add_scalar("train/loss_obj_risk", tr["loss_obj_risk"], epoch)
            writer.add_scalar("train/acc",           tr["acc"] * 100,     epoch)
            writer.add_scalar("train/lr",            lr_now,              epoch)
            writer.add_scalar("train/time_sec",      tr["time"],          epoch)
            if "vram_peak_gb" in vram_tr:
                writer.add_scalar("sys/vram_peak_gb",
                                  vram_tr["vram_peak_gb"], epoch)

        # ── Validation ────────────────────────────────────────
        val_stats = {}
        if "val" in loaders:
            val_stats = evaluate(model, loaders["val"], device, "val", use_amp)
            log_vram("val")

        # ── TensorBoard val ───────────────────────────────────
        if writer and val_stats:
            writer.add_scalar("val/acc",
                              val_stats.get("val_acc", 0), epoch)
            writer.add_scalar("val/obj_risk_acc",
                              val_stats.get("val_obj_risk_acc", 0), epoch)
            for d_name, acc in val_stats.get("val_domain_acc", {}).items():
                writer.add_scalar(f"val/acc_{d_name}", acc, epoch)
            for r_name, acc in val_stats.get("val_class_acc", {}).items():
                writer.add_scalar(f"val/acc_{r_name}", acc, epoch)
            for r_name, acc in val_stats.get("val_obj_class_acc", {}).items():
                writer.add_scalar(f"val/obj_acc_{r_name}", acc, epoch)
            for d_name, unc in val_stats.get("val_unc", {}).items():
                writer.add_scalar(f"val/unc_{d_name}", unc, epoch)
            writer.flush()

        # ── Logging ───────────────────────────────────────────
        epoch_log = {"epoch": epoch, **tr, **vram_tr, **val_stats}
        history.append(epoch_log)

        # ── Save best ─────────────────────────────────────────
        val_acc = val_stats.get("val_acc", 0.0)
        if val_acc > best_val:
            best_val      = val_acc
            best_ep       = epoch
            epochs_no_impr = 0
            raw_model = (model._orig_mod
                         if hasattr(model, "_orig_mod") else model)
            torch.save({
                "epoch":       epoch,
                "model_state": raw_model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "scaler":      scaler.state_dict(),
                "val_acc":     val_acc,
                "cfg":         cfg,
            }, save_dir / "best_model.pth")
            print(f"  ✓ Best model disimpan "
                  f"(val_acc={val_acc:.2f}%, epoch={epoch})")
        else:
            epochs_no_impr += 1

        # ── Checkpoint berkala ────────────────────────────────
        if epoch % cfg["save_every"] == 0:
            raw_model = (model._orig_mod
                         if hasattr(model, "_orig_mod") else model)
            torch.save({
                "epoch":       epoch,
                "model_state": raw_model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "scaler":      scaler.state_dict(),
                "val_acc":     val_acc,
                "cfg":         cfg,
            }, save_dir / f"ckpt_ep{epoch:03d}.pth")

        # ── Early stopping ────────────────────────────────────
        if epochs_no_impr >= patience:
            print(f"\n  Early stopping di epoch {epoch} "
                  f"(tidak ada improvement selama {patience} epoch)")
            break

    if writer:
        writer.close()

    # ── Final test ────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Training selesai.  Best: epoch={best_ep}, "
          f"val_acc={best_val:.2f}%")

    if "test" in loaders:
        print("\nEvaluasi test set dengan best model...")
        ckpt      = torch.load(save_dir / "best_model.pth", map_location=device)
        raw_model = (model._orig_mod if hasattr(model, "_orig_mod") else model)
        raw_model.load_state_dict(ckpt["model_state"])
        test_stats = evaluate(model, loaders["test"], device, "test", use_amp)
        log_vram("test")
        history.append({"final_test": test_stats})
        print(f"\n  [TEST]  acc={test_stats['test_acc']:.2f}%  "
              f"obj_risk_acc={test_stats['test_obj_risk_acc']:.2f}%")

    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory → {save_dir / 'history.json'}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train DASTRP-Net — RTX 5000 Ada optimized"
    )
    parser.add_argument("--dataset",     default=DEFAULT_CFG["dataset_root"])
    parser.add_argument("--yolo",        required=True)
    parser.add_argument("--yolo-conf",   type=float, default=0.25)
    parser.add_argument("--yolo-iou",    type=float, default=0.45)
    parser.add_argument("--T",           type=int,   default=4)
    parser.add_argument("--stride",      type=int,   default=2)
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch",       type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--img-size",    type=int,   default=640)
    parser.add_argument("--workers",     type=int,   default=8)
    parser.add_argument("--save-dir",    default="runs/exp1")
    parser.add_argument("--save-every",  type=int,   default=10)
    parser.add_argument("--patience",    type=int,   default=20)
    parser.add_argument("--resume",      default=None)
    parser.add_argument("--T-agg",       type=int,   default=None)
    parser.add_argument("--no-gt-boxes", action="store_true")
    parser.add_argument("--no-amp",      action="store_true")
    parser.add_argument("--no-compile",  action="store_true")
    add_seed_args(parser)
    parser.add_argument(
        "--ablation", default=None,
        choices=["no_gnn", "no_cbn", "no_fusion",
                 "no_edl", "no_velocity", "no_obj_risk"],
        help="Ablation skenario (default: None = full model)"
    )
    args = parser.parse_args()

    cfg = {
        **DEFAULT_CFG,
        "dataset_root": args.dataset,
        "yolo_weights": args.yolo,
        "yolo_conf":    args.yolo_conf,
        "yolo_iou":     args.yolo_iou,
        "T":            args.T,
        "stride":       args.stride,
        "epochs":       args.epochs,
        "batch_size":   args.batch,
        "lr":           args.lr,
        "img_size":     args.img_size,
        "num_workers":  args.workers,
        "save_dir":     args.save_dir,
        "save_every":   args.save_every,
        "patience":     args.patience,
        "resume":       args.resume,
        "T_agg":        args.T_agg,
        "use_gt_boxes": not args.no_gt_boxes,
        "use_amp":      not args.no_amp,
        "use_compile":  not args.no_compile,
        "ablation":     args.ablation,
        "seed":         args.seed,
        "deterministic": args.deterministic,
    }

    train(cfg)
