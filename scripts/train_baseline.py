"""
scripts/train_baseline.py
=========================
Training script untuk baseline models B1, B2, B3.
Menggunakan train.py yang sama tapi model factory berbeda.

Cara pakai:
  python scripts/train_baseline.py \
    --model   b1 \
    --dataset /path/to/dataset \
    --yolo    /path/to/best.pt \
    --save-dir runs/baseline_b1

Semua argumen lain identik dengan train.py.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.reproducibility import set_seed, add_seed_args


def build_model(model_name: str, cfg: dict, device):
    """Factory function untuk membuat model baseline."""

    if model_name == "b1":
        from models.baseline_b1 import BaselineB1
        model = BaselineB1(
            yolo_weights = cfg["yolo_weights"],
            yolo_conf    = cfg.get("yolo_conf", 0.25),
            yolo_iou     = cfg.get("yolo_iou",  0.45),
            T_agg        = cfg.get("T_agg",     None),
        ).to(device)
        print(f"[Baseline B1] YOLO11n + MLP (frame-only, no temporal)")

    elif model_name == "b2":
        from models.baseline_b2 import BaselineB2
        model = BaselineB2(
            yolo_weights = cfg["yolo_weights"],
            yolo_conf    = cfg.get("yolo_conf",    0.25),
            yolo_iou     = cfg.get("yolo_iou",     0.45),
            lstm_dim     = cfg.get("lstm_dim",     256),
            lstm_layers  = cfg.get("lstm_layers",  2),
            lstm_dropout = cfg.get("lstm_dropout", 0.3),
            T_agg        = cfg.get("T_agg",        None),
        ).to(device)
        print(f"[Baseline B2] YOLO11n + BiLSTM "
              f"(dim={cfg.get('lstm_dim',256)}, "
              f"layers={cfg.get('lstm_layers',2)})")

    elif model_name == "b3":
        from models.baseline_b3 import BaselineB3
        model = BaselineB3(
            yolo_weights  = cfg["yolo_weights"],
            yolo_conf     = cfg.get("yolo_conf",    0.25),
            yolo_iou      = cfg.get("yolo_iou",     0.45),
            tf_n_heads    = cfg.get("tf_n_heads",   4),
            tf_n_layers   = cfg.get("tf_n_layers",  2),
            tf_dim_ff     = cfg.get("tf_dim_ff",    512),
            tf_dropout    = cfg.get("tf_dropout",   0.1),
            tf_pool_mode  = cfg.get("tf_pool_mode", "cls"),
            T_agg         = cfg.get("T_agg",        None),
        ).to(device)
        print(f"[Baseline B3] YOLO11n + Transformer "
              f"(heads={cfg.get('tf_n_heads',4)}, "
              f"layers={cfg.get('tf_n_layers',2)}, "
              f"pool={cfg.get('tf_pool_mode','cls')})")

    elif model_name == "b4":
        from models.baseline_b4 import BaselineB4
        model = BaselineB4(
            yolo_weights = cfg["yolo_weights"],
            yolo_conf    = cfg.get("yolo_conf",    0.25),
            yolo_iou     = cfg.get("yolo_iou",     0.45),
            gcn_hidden   = cfg.get("gcn_hidden",   256),
            gcn_layers   = cfg.get("gcn_layers",   2),
            gcn_dropout  = cfg.get("gcn_dropout",  0.3),
            lstm_dim     = cfg.get("sg_lstm_dim",  256),
            lstm_layers  = cfg.get("sg_lstm_layers", 1),
            lstm_dropout = cfg.get("sg_lstm_dropout", 0.2),
            T_agg        = cfg.get("T_agg",        None),
        ).to(device)
        print(f"[Baseline B4] Scene-Graph GCN + LSTM "
              f"(gcn_layers={cfg.get('gcn_layers',2)}, "
              f"lstm_dim={cfg.get('sg_lstm_dim',256)})")

    elif model_name == "b5":
        from models.baseline_b5 import BaselineB5
        model = BaselineB5(
            base_channels_slow = cfg.get("sf_base_ch",   64),
            alpha              = cfg.get("sf_alpha",      4),
            dropout            = cfg.get("sf_dropout",    0.5),
        ).to(device)
        print(f"[Baseline B5] SlowFast "
              f"(alpha={cfg.get('sf_alpha',4)}, "
              f"base_ch={cfg.get('sf_base_ch',64)})")

    elif model_name == "dastrp":
        from models.architecture import DASTRPNet
        model = DASTRPNet(
            yolo_weights = cfg["yolo_weights"],
            yolo_conf    = cfg.get("yolo_conf", 0.25),
            yolo_iou     = cfg.get("yolo_iou",  0.45),
            dist_thresh  = cfg.get("dist_thresh", 0.3),
            T_agg        = cfg.get("T_agg", None),
        ).to(device)
        print(f"[Full DASTRP-Net]")

    else:
        raise ValueError(f"Model tidak dikenal: {model_name}. "
                         f"Pilihan: b1, b2, dastrp")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params : {n_train:,}")
    if hasattr(model, "yolo"):
        n_yolo = sum(p.numel() for p in model.yolo.parameters())
        print(f"  Frozen (YOLO)    : {n_yolo:,}")
    else:
        print(f"  Frozen (YOLO)    : N/A (detection-free model)")
    return model


def get_b1_loss_cfg(cfg: dict) -> dict:
    """
    Override loss weights untuk B1:
      - lambda_domain = 0   → domain_logits adalah dummy
      - lambda_motion = 0   → tidak ada motion head
      - lambda_obj_risk = 0 → tidak ada per-objek detection
    Loss aktif hanya: EDL agg + per-frame CE
    """
    b1_cfg = dict(cfg)
    b1_cfg["lambda_domain"]   = 0.0
    b1_cfg["lambda_motion"]   = 0.0
    b1_cfg["lambda_obj_risk"] = 0.0
    b1_cfg["lambda_unc"]      = 0.0   # tidak ada uncertainty head nyata
    return b1_cfg


def get_b2_loss_cfg(cfg: dict) -> dict:
    """
    Override loss weights untuk B2:
    Sama dengan B1 — domain/motion/obj_risk adalah dummy.
    Per-frame CE tetap aktif karena BiLSTM hidden states
    dipakai untuk all_frame_risks.
    """
    b2_cfg = dict(cfg)
    b2_cfg["lambda_domain"]   = 0.0
    b2_cfg["lambda_motion"]   = 0.0
    b2_cfg["lambda_obj_risk"] = 0.0
    b2_cfg["lambda_unc"]      = 0.0
    return b2_cfg


def get_b3_loss_cfg(cfg: dict) -> dict:
    """Override loss weights untuk B3 — identik dengan B1/B2."""
    b3_cfg = dict(cfg)
    b3_cfg["lambda_domain"]   = 0.0
    b3_cfg["lambda_motion"]   = 0.0
    b3_cfg["lambda_obj_risk"] = 0.0
    b3_cfg["lambda_unc"]      = 0.0
    return b3_cfg


def get_b4_loss_cfg(cfg: dict) -> dict:
    """Override loss weights untuk B4 — sama dengan B1/B2/B3."""
    b4_cfg = dict(cfg)
    b4_cfg["lambda_domain"]   = 0.0
    b4_cfg["lambda_motion"]   = 0.0
    b4_cfg["lambda_obj_risk"] = 0.0
    b4_cfg["lambda_unc"]      = 0.0
    b4_cfg["use_gt_boxes"]    = True
    return b4_cfg


def get_b5_loss_cfg(cfg: dict) -> dict:
    """
    Override loss weights untuk B5 (SlowFast).
    B5 tidak menggunakan YOLO sama sekali, sehingga
    use_gt_boxes tidak relevan (tidak ada objek).
    Per-frame CE aktif karena all_frame_risks diisi dari
    replikasi prediksi akhir (bukan zero).
    """
    b5_cfg = dict(cfg)
    b5_cfg["lambda_domain"]   = 0.0
    b5_cfg["lambda_motion"]   = 0.0
    b5_cfg["lambda_obj_risk"] = 0.0
    b5_cfg["lambda_unc"]      = 0.0
    b5_cfg["use_gt_boxes"]    = False  # B5 tidak pakai detection
    return b5_cfg


def main():
    parser = argparse.ArgumentParser(
        description="Train DASTRP-Net baseline models (B1, B2)"
    )
    parser.add_argument("--model",        default="b1",
                        choices=["b1", "b2", "b3", "b4", "b5", "dastrp"],
                        help="Model yang ditraining")
    parser.add_argument("--dataset",      required=True)
    parser.add_argument("--yolo",         required=True)
    parser.add_argument("--yolo-conf",    type=float, default=0.25)
    parser.add_argument("--yolo-iou",     type=float, default=0.45)
    parser.add_argument("--T",            type=int,   default=4)
    parser.add_argument("--stride",       type=int,   default=2)
    parser.add_argument("--epochs",       type=int,   default=50)
    parser.add_argument("--batch",        type=int,   default=16)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--img-size",     type=int,   default=640)
    parser.add_argument("--workers",      type=int,   default=4)
    parser.add_argument("--save-dir",     default="runs/baseline_b1")
    parser.add_argument("--save-every",   type=int,   default=10)
    parser.add_argument("--patience",     type=int,   default=15)
    parser.add_argument("--resume",       default=None)
    add_seed_args(parser)
    parser.add_argument("--no-amp",       action="store_true")
    # B2-specific args
    parser.add_argument("--lstm-dim",     type=int,   default=256,
                        help="BiLSTM hidden dim per direction (B2 only)")
    parser.add_argument("--lstm-layers",  type=int,   default=2,
                        help="Jumlah layer BiLSTM (B2 only)")
    parser.add_argument("--lstm-dropout", type=float, default=0.3,
                        help="Dropout antar LSTM layers (B2 only)")
    # B3-specific args
    parser.add_argument("--tf-n-heads",   type=int,   default=4,
                        help="Jumlah attention heads Transformer (B3 only)")
    parser.add_argument("--tf-n-layers",  type=int,   default=2,
                        help="Jumlah Transformer encoder layers (B3 only)")
    parser.add_argument("--tf-dim-ff",    type=int,   default=512,
                        help="FFN inner dim Transformer (B3 only)")
    parser.add_argument("--tf-dropout",   type=float, default=0.1,
                        help="Dropout Transformer (B3 only)")
    parser.add_argument("--tf-pool-mode", type=str,   default="cls",
                        choices=["cls", "mean"],
                        help="Aggregation mode: cls token atau mean pool (B3 only)")
    # B4-specific args
    parser.add_argument("--gcn-hidden",      type=int,   default=256,
                        help="GCN hidden dim (B4 only)")
    parser.add_argument("--gcn-layers",      type=int,   default=2,
                        help="Jumlah GCN layers (B4 only)")
    parser.add_argument("--gcn-dropout",     type=float, default=0.3,
                        help="GCN dropout (B4 only)")
    parser.add_argument("--sg-lstm-dim",     type=int,   default=256,
                        help="LSTM hidden dim untuk B4 Scene-Graph")
    parser.add_argument("--sg-lstm-layers",  type=int,   default=1,
                        help="Jumlah LSTM layers untuk B4")
    parser.add_argument("--sg-lstm-dropout", type=float, default=0.2,
                        help="LSTM dropout untuk B4")
    # B5-specific args
    parser.add_argument("--sf-base-ch",  type=int,   default=64,
                        help="Base channel Slow pathway (B5 only)")
    parser.add_argument("--sf-alpha",    type=int,   default=4,
                        help="Temporal stride ratio T/T_slow (B5 only)")
    parser.add_argument("--sf-dropout",  type=float, default=0.5,
                        help="Dropout sebelum risk head (B5 only)")
    args = parser.parse_args()

    # Import fungsi training dari train.py yang sudah ada
    from scripts.train import (
        evaluate, compute_class_weights,
        make_scheduler, log_vram, MetricTracker,
        compute_loss
    )
    from data.dataset import get_dataloaders

    import json, math, time, torch, torch.nn as nn, torch.optim as optim
    from torch.utils.tensorboard import SummaryWriter
    from tqdm import tqdm

    # ── Safe train_epoch — handle model tanpa yolo attribute ──────
    def safe_train_epoch(
        model, loader, optimizer, scheduler, scaler,
        epoch, num_epochs, device, cfg, class_weights=None,
    ):
        """
        Sama dengan train_epoch di train.py tapi aman untuk
        model tanpa self.yolo (B5 SlowFast).
        """
        model.train()
        # Freeze YOLO backbone jika ada (B1-B4), skip jika tidak (B5)
        if hasattr(model, "yolo") and hasattr(model.yolo, "backbone"):
            model.yolo.backbone.eval()

        tracker  = MetricTracker()
        t0       = time.time()
        use_amp  = cfg["use_amp"] and device.type == "cuda"

        pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{num_epochs}",
                    unit="batch", dynamic_ncols=True, leave=False)

        for step, batch in enumerate(pbar):
            clip_images  = batch["images"].to(device, non_blocking=True)
            domain_ids   = batch["domain_ids"].to(device, non_blocking=True)
            gt_boxes_seq = (batch["boxes_seq"]
                            if cfg.get("use_gt_boxes") else None)

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
                    outputs, batch, epoch, num_epochs,
                    cfg, device, class_weights
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

            pred_risk = outputs["risk_pred"]
            true_risk = batch["frame_risks"].max(dim=1).values.to(device)
            acc       = (pred_risk == true_risk).float().mean()

            tracker.update(
                loss     = losses["total"].detach(),
                loss_edl = losses["edl_agg"],
                acc      = acc,
            )
            pbar.set_postfix({
                "loss": f"{tracker.mean('loss'):.4f}",
                "acc":  f"{tracker.mean('acc')*100:.1f}%",
                "lr":   f"{scheduler.get_last_lr()[0]:.1e}",
            }, refresh=False)

        summary         = tracker.summary()
        summary["time"] = round(time.time() - t0, 1)
        return summary

    # ── Seed ──────────────────────────────────────────────────────
    set_seed(args.seed, args.deterministic)

    # ── Config ────────────────────────────────────────────────────
    cfg = {
        "dataset_root":    args.dataset,
        "yolo_weights":    args.yolo,
        "yolo_conf":       args.yolo_conf,
        "yolo_iou":        args.yolo_iou,
        "T":               args.T,
        "stride":          args.stride,
        "epochs":          args.epochs,
        "batch_size":      args.batch,
        "lr":              args.lr,
        "img_size":        args.img_size,
        "num_workers":     args.workers,
        "save_dir":        args.save_dir,
        "save_every":      args.save_every,
        "patience":        args.patience,
        "resume":          args.resume,
        "seed":            args.seed,
        "deterministic":   args.deterministic,
        "use_amp":         not args.no_amp,
        "use_compile":     False,
        "use_gt_boxes":    True,
        "T_agg":           None,
        "warmup_epochs":   2,
        "weight_decay":    1e-4,
        "grad_clip":       5.0,
        "model":           args.model,
        # B2 BiLSTM hyperparameters
        "lstm_dim":        args.lstm_dim,
        "lstm_layers":     args.lstm_layers,
        "lstm_dropout":    args.lstm_dropout,
        # B3 Transformer hyperparameters
        "tf_n_heads":      args.tf_n_heads,
        "tf_n_layers":     args.tf_n_layers,
        "tf_dim_ff":       args.tf_dim_ff,
        "tf_dropout":      args.tf_dropout,
        "tf_pool_mode":    args.tf_pool_mode,
        # B4 Scene-Graph GCN + LSTM hyperparameters
        "gcn_hidden":        args.gcn_hidden,
        "gcn_layers":        args.gcn_layers,
        "gcn_dropout":       args.gcn_dropout,
        "sg_lstm_dim":       args.sg_lstm_dim,
        "sg_lstm_layers":    args.sg_lstm_layers,
        "sg_lstm_dropout":   args.sg_lstm_dropout,
        # B5 SlowFast hyperparameters
        "sf_base_ch":        args.sf_base_ch,
        "sf_alpha":          args.sf_alpha,
        "sf_dropout":        args.sf_dropout,
        # Loss weights default (akan di-override per model)
        "lambda_frame":    1.0,
        "lambda_domain":   0.2,
        "lambda_motion":   0.2,
        "lambda_obj_risk": 0.3,
        "lambda_unc":      0.1,
        "unc_target":      0.4,
    }

    # Override loss weights sesuai model
    if args.model == "b1":
        cfg = get_b1_loss_cfg(cfg)
    elif args.model == "b2":
        cfg = get_b2_loss_cfg(cfg)
    elif args.model == "b3":
        cfg = get_b3_loss_cfg(cfg)
    elif args.model == "b4":
        cfg = get_b4_loss_cfg(cfg)
    elif args.model == "b5":
        cfg = get_b5_loss_cfg(cfg)

    # ── Setup ─────────────────────────────────────────────────────
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not cfg.get("deterministic", False):
        torch.backends.cudnn.benchmark     = True
        torch.backends.cudnn.deterministic = False

    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    writer = SummaryWriter(log_dir=str(save_dir / "tensorboard"))

    print("=" * 65)
    print(f"BASELINE TRAINING — {args.model.upper()}")
    print("=" * 65)
    print(f"Device  : {device}  |  AMP: {'on' if cfg['use_amp'] else 'off'}")
    print(f"Dataset : {cfg['dataset_root']}")
    print(f"T={cfg['T']}, stride={cfg['stride']}, batch={cfg['batch_size']}")
    print(f"Epochs  : {cfg['epochs']}  |  patience={cfg['patience']}")
    if args.model == "b2":
        print(f"BiLSTM  : dim={cfg['lstm_dim']}, "
              f"layers={cfg['lstm_layers']}, "
              f"dropout={cfg['lstm_dropout']}")
    elif args.model == "b3":
        print(f"Transformer: heads={cfg['tf_n_heads']}, "
              f"layers={cfg['tf_n_layers']}, "
              f"ff={cfg['tf_dim_ff']}, "
              f"pool={cfg['tf_pool_mode']}, "
              f"dropout={cfg['tf_dropout']}")
    elif args.model == "b4":
        print(f"SceneGraph: gcn_hidden={cfg['gcn_hidden']}, "
              f"gcn_layers={cfg['gcn_layers']}, "
              f"lstm_dim={cfg['sg_lstm_dim']}, "
              f"lstm_layers={cfg['sg_lstm_layers']}")
    elif args.model == "b5":
        print(f"SlowFast: alpha={cfg['sf_alpha']}, "
              f"base_ch={cfg['sf_base_ch']}, "
              f"dropout={cfg['sf_dropout']}")
    print(f"Active loss weights:")
    for k in ["lambda_frame","lambda_domain","lambda_motion",
              "lambda_obj_risk","lambda_unc"]:
        v = cfg[k]
        status = "" if v > 0 else "  ← disabled"
        print(f"  {k}: {v}{status}")
    print()

    # ── DataLoaders ───────────────────────────────────────────────
    loaders = get_dataloaders(
        dataset_root = cfg["dataset_root"],
        T            = cfg["T"],
        stride       = cfg["stride"],
        img_size     = cfg["img_size"],
        batch_size   = cfg["batch_size"],
        num_workers  = cfg["num_workers"],
        seed         = cfg["seed"],
        splits       = ["train", "val"],
    )
    assert "train" in loaders

    class_weights = compute_class_weights(loaders, device)

    # ── Model ─────────────────────────────────────────────────────
    model = build_model(args.model, cfg, device)

    # ── AMP Scaler ────────────────────────────────────────────────
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(cfg["use_amp"] and device.type == "cuda")
    )

    # ── Optimizer ─────────────────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable,
        lr           = cfg["lr"],
        weight_decay = cfg["weight_decay"],
        fused        = (device.type == "cuda"),
    )
    scheduler = make_scheduler(optimizer, cfg, len(loaders["train"]))

    # ── Resume ────────────────────────────────────────────────────
    start_epoch    = 1
    best_val       = 0.0
    best_ep        = 0
    epochs_no_impr = 0

    if cfg.get("resume") and Path(cfg["resume"]).exists():
        ckpt = torch.load(cfg["resume"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val    = ckpt.get("val_acc", 0.0)
        best_ep     = ckpt.get("epoch", 0)
        print(f"Resumed dari epoch {start_epoch}, best_val={best_val:.2f}%")

    log_vram("sebelum training")
    use_amp = cfg["use_amp"] and device.type == "cuda"
    history = []

    # ── Training loop ─────────────────────────────────────────────
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        lr_now = scheduler.get_last_lr()[0]
        print(f"\n{'='*65}")
        print(f"Epoch {epoch}/{cfg['epochs']}  LR={lr_now:.2e}  "
              f"[{args.model.upper()}]")
        print("=" * 65)

        tr = safe_train_epoch(
            model, loaders["train"], optimizer, scheduler, scaler,
            epoch, cfg["epochs"], device, cfg, class_weights,
        )
        vram_tr = log_vram("train")

        print(f"\n  [TRAIN]  loss={tr['loss']:.4f}  "
              f"acc={tr['acc']*100:.2f}%  t={tr['time']}s")

        # TensorBoard
        writer.add_scalar("train/loss",  tr["loss"],      epoch)
        writer.add_scalar("train/acc",   tr["acc"] * 100, epoch)
        writer.add_scalar("train/lr",    lr_now,          epoch)

        # ── Validation ────────────────────────────────────────────
        val_stats = {}
        if "val" in loaders:
            val_stats = evaluate(model, loaders["val"], device, "val", use_amp)
            log_vram("val")
            writer.add_scalar("val/acc",      val_stats.get("val_acc", 0), epoch)
            writer.add_scalar("val/macro_f1",
                              val_stats.get("val_f1", {}).get("macro", {}).get("f1", 0),
                              epoch)
            for r, acc in val_stats.get("val_class_acc", {}).items():
                writer.add_scalar(f"val/acc_{r}", acc, epoch)
            writer.flush()

        # ── Logging ───────────────────────────────────────────────
        epoch_log = {"epoch": epoch, **tr, **vram_tr, **val_stats}
        history.append(epoch_log)

        # ── Save best ─────────────────────────────────────────────
        val_acc = val_stats.get("val_acc", 0.0)
        if val_acc > best_val:
            best_val       = val_acc
            best_ep        = epoch
            epochs_no_impr = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "scaler":      scaler.state_dict(),
                "val_acc":     val_acc,
                "cfg":         cfg,
                "model_name":  args.model,
            }, save_dir / "best_model.pth")
            print(f"  ✓ Best saved (val_acc={val_acc:.2f}%, ep={epoch})")
        else:
            epochs_no_impr += 1

        if epoch % cfg["save_every"] == 0:
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optimizer":   optimizer.state_dict(),
                "scaler":      scaler.state_dict(),
                "val_acc":     val_acc,
                "cfg":         cfg,
                "model_name":  args.model,
            }, save_dir / f"ckpt_ep{epoch:03d}.pth")

        # ── Early stopping ────────────────────────────────────────
        if epochs_no_impr >= cfg["patience"]:
            print(f"\n  Early stopping di epoch {epoch} "
                  f"(no improvement {cfg['patience']} epoch)")
            break

    writer.close()

    # ── Final test ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"Selesai. Best: epoch={best_ep}, val_acc={best_val:.2f}%")

    print("\nEvaluasi test set...")
    test_loaders = get_dataloaders(
        dataset_root = cfg["dataset_root"],
        T            = cfg["T"],
        stride       = cfg["T"],   # non-overlapping
        img_size     = cfg["img_size"],
        batch_size   = cfg["batch_size"],
        num_workers  = cfg["num_workers"],
        seed         = cfg["seed"],
        splits       = ["test"],
    )
    ckpt = torch.load(save_dir / "best_model.pth", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_stats = evaluate(model, test_loaders["test"], device, "test", use_amp)
    history.append({"final_test": test_stats})
    print(f"\n  [TEST] acc={test_stats['test_acc']:.2f}%  "
          f"macro_f1={test_stats.get('test_f1',{}).get('macro',{}).get('f1',0):.2f}%")

    import json
    with open(save_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nHistory → {save_dir / 'history.json'}")


if __name__ == "__main__":
    main()
