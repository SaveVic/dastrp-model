"""
scripts/inference.py
====================
Inference DASTRP-Net pada video — berbasis clip (T frame berurutan).

Model menerima clip (1, T, 3, H, W) per forward call, bukan frame tunggal.
Frame diakumulasi dalam buffer sebelum diproses, sehingga:
  - T-1 frame pertama video: buffer belum penuh, output dari partial clip
  - Frame selanjutnya: sliding window bergeser stride=1 frame
    (setiap frame baru langsung di-predict dengan konteks T-1 frame sebelumnya)

Output per frame:
  - Annotated video (opsional)
  - CSV log: frame_id, risk_id, risk_label, uncertainty, n_det,
             p_safe, p_warn, p_danger, domain_pred, n_gerak, n_diam

Cara pakai:
  python inference.py \\
    --checkpoint runs/exp1/best_model.pth \\
    --yolo       /path/to/best.pt \\
    --input      /path/to/video.mp4 \\
    --output     output/annotated.mp4 \\
    --log-csv    output/log.csv \\
    --T          4
"""

import cv2
import csv
import torch
import numpy as np
from collections import deque
from pathlib import Path
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.reproducibility import set_seed, add_seed_args
from models.architecture import DASTRPNet
from data.dataset import infer_domain, DOMAIN_NAMES, RISK_NAMES

# Label warna per risk level (BGR untuk OpenCV)
RISK_COLOR = {
    0: (80,  200, 50),    # safe    — hijau
    1: (30,  180, 255),   # warning — kuning
    2: (60,   60, 255),   # danger  — merah
}
RISK_BG = {
    0: (20,  60, 20),
    1: (10,  60, 80),
    2: (20,  10, 80),
}
MOTION_NAMES = {0: "gerak", 1: "diam"}


# ─────────────────────────────────────────────
# Inferencer
# ─────────────────────────────────────────────

class DASTRPInferencer:
    """
    Wrapper inference DASTRP-Net untuk video streaming.

    Menggunakan sliding window buffer T frame.
    Setiap frame masuk diproses sebagai bagian dari clip terakhir.
    VelocityTracker di-reset setiap kali buffer dikosongkan (awal video).
    """

    def __init__(
        self,
        checkpoint:  str,
        yolo_weights: str,
        T:           int   = 4,
        img_size:    int   = 640,
        device:      str   = None,
    ):
        self.T        = T
        self.img_size = img_size
        self.device   = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # ── Load model ────────────────────────────────────────
        print(f"Loading checkpoint : {checkpoint}")
        ckpt = torch.load(checkpoint, map_location=self.device)
        cfg  = ckpt.get("cfg", {})

        self.model = DASTRPNet(
            yolo_weights = yolo_weights or cfg.get("yolo_weights", ""),
            yolo_conf    = cfg.get("yolo_conf", 0.25),
            yolo_iou     = cfg.get("yolo_iou",  0.45),
            dist_thresh  = cfg.get("dist_thresh", 0.3),
            T_agg        = cfg.get("T_agg", None),
            ablation     = cfg.get("ablation", None),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"Model loaded.  Val acc : {ckpt.get('val_acc', 'N/A')}%")
        print(f"Clip length T = {T}")

        # ── Transform ─────────────────────────────────────────
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225],
            ),
        ])

        # ── Sliding window buffer ──────────────────────────────
        # Menyimpan T frame tensor terakhir
        self._buffer: deque = deque(maxlen=T)

        # Tracker sudah di-reset di dalam model.forward() setiap call
        # Tapi kita perlu reset manual saat buffer pertama kali diisi
        self._buffer_full = False

    def reset(self):
        """Reset buffer dan tracker. Panggil setiap awal video baru."""
        self._buffer.clear()
        self._buffer_full = False
        # Tracker di dalam model akan di-reset di forward() call berikutnya

    def _preprocess(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """BGR frame → normalized tensor (3, H, W)."""
        rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.img_size, self.img_size))
        return self.transform(resized)   # (3, H, W)

    @torch.no_grad()
    def predict_frame(
        self,
        frame_bgr: np.ndarray,
        domain_id: int = None,
    ) -> dict:
        """
        Proses satu frame video.

        Frame dimasukkan ke buffer. Begitu buffer penuh (T frame),
        clip dikirim ke model. Jika buffer belum penuh, clip di-pad
        dengan frame yang tersedia (replikasi frame awal).

        Args:
            frame_bgr : frame BGR dari OpenCV
            domain_id : 0=UGM, 1=Utara, 2=Ringroad (None=auto dari scene clf)

        Returns:
            dict berisi risk_id, risk_prob, uncertainty, boxes, motion_pred, dsb.
        """
        img_t = self._preprocess(frame_bgr)   # (3, H, W)
        self._buffer.append(img_t)

        # Susun clip dari buffer (pad dengan frame awal jika belum penuh)
        frames = list(self._buffer)
        if len(frames) < self.T:
            # Pad: replikasi frame pertama di depan
            pad = [frames[0]] * (self.T - len(frames))
            frames = pad + frames

        # Clip tensor: (1, T, 3, H, W)
        clip = torch.stack(frames, dim=0).unsqueeze(0).to(self.device)

        dom_t = (torch.tensor([domain_id], dtype=torch.long, device=self.device)
                 if domain_id is not None else None)

        outputs = self.model(
            clip_images  = clip,
            domain_ids   = dom_t,
            gt_boxes_seq = None,   # inference: pakai YOLO
        )

        # Ekstrak output frame terakhir
        boxes_last  = outputs["det_boxes"][0]       # (N, 4)
        motion_last = outputs["motion_logits"][0]   # (N, 2)

        motion_pred = (motion_last.argmax(1).cpu().numpy()
                       if motion_last.size(0) > 0 else np.array([]))

        # Per-objek risk prediksi
        obj_risk_last = outputs["obj_risk_pred"][0]
        obj_risk_pred = (obj_risk_last.cpu().numpy()
                         if obj_risk_last.size(0) > 0 else np.array([]))

        return {
            "risk_id":      outputs["risk_pred"][0].item(),
            "risk_prob":    outputs["risk_prob"][0].cpu().numpy(),   # (3,)
            "uncertainty":  outputs["uncertainty"][0].item(),
            "domain_pred":  outputs["domain_logits"][0].argmax().item(),
            "alpha":        outputs["alpha"][0].cpu().numpy(),       # (3,)
            "boxes":        boxes_last.cpu().numpy(),                # (N, 4)
            "motion_pred":  motion_pred,                             # (N,)
            "obj_risk_pred": obj_risk_pred,                          # (N,)
        }

    # ── Annotation ────────────────────────────────────────────

    def annotate(
        self,
        frame_bgr: np.ndarray,
        result:    dict,
        show_prob: bool = True,
    ) -> np.ndarray:
        """
        Gambar bounding box + risk overlay + uncertainty bar.

        Args:
            frame_bgr  : frame asli (1280×720 BGR)
            result     : output dari predict_frame()
            show_prob  : tampilkan probabilitas 3 kelas

        Returns:
            frame ter-anotasi (BGR)
        """
        out   = frame_bgr.copy()
        H, W  = out.shape[:2]
        r     = result["risk_id"]
        color = RISK_COLOR[r]
        bg    = RISK_BG[r]
        unc   = result["uncertainty"]
        prob  = result["risk_prob"]
        boxes = result["boxes"]
        mots  = result["motion_pred"]

        # ── Bounding boxes — warna per-objek risk ─────────────
        obj_risks = result.get("obj_risk_pred", [])
        n_gerak = n_diam = 0
        for i, box in enumerate(boxes):
            cx, cy, bw, bh = box
            x1 = max(0, int((cx - bw / 2) * W))
            y1 = max(0, int((cy - bh / 2) * H))
            x2 = min(W, int((cx + bw / 2) * W))
            y2 = min(H, int((cy + bh / 2) * H))

            # Warna box dari per-objek risk (jika tersedia),
            # fallback ke warna frame-level risk
            if i < len(obj_risks):
                obj_r     = int(obj_risks[i])
                box_color = RISK_COLOR[obj_r]
                obj_lbl   = RISK_NAMES[obj_r][0]   # S / W / D
            else:
                box_color = color
                obj_lbl   = ""

            cv2.rectangle(out, (x1, y1), (x2, y2), box_color, 2)

            # Label: motion + per-objek risk
            parts = []
            if i < len(mots):
                mot_id  = int(mots[i])
                mot_lbl = MOTION_NAMES.get(mot_id, "?")
                parts.append(mot_lbl)
                if mot_id == 0:
                    n_gerak += 1
                else:
                    n_diam  += 1
            if obj_lbl:
                parts.append(obj_lbl)

            if parts:
                cv2.putText(out, "|".join(parts),
                            (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, box_color, 1,
                            cv2.LINE_AA)

        # ── Panel kiri atas: risk level ───────────────────────
        panel_w, panel_h = 310, 92
        overlay = out.copy()
        cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), bg, -1)
        out = cv2.addWeighted(overlay, 0.6, out, 0.4, 0)

        cv2.putText(out, f"RISK: {RISK_NAMES[r]}",
                    (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2,
                    cv2.LINE_AA)

        # Uncertainty bar
        bar_x1, bar_y1 = 18, 54
        bar_x2, bar_y2 = 300, 70
        cv2.rectangle(out, (bar_x1, bar_y1), (bar_x2, bar_y2),
                      (50, 50, 50), -1)
        unc_w = int((bar_x2 - bar_x1) * unc)
        unc_c = (255, 120, 50) if unc >= 0.4 else (100, 200, 255)
        cv2.rectangle(out, (bar_x1, bar_y1),
                      (bar_x1 + unc_w, bar_y2), unc_c, -1)
        cv2.putText(out, f"Uncertainty: {unc:.3f}",
                    (18, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (200, 200, 200), 1, cv2.LINE_AA)

        # ── Panel kanan atas: info ────────────────────────────
        dom_name = DOMAIN_NAMES.get(result["domain_pred"], "?")
        info_lines = [
            f"Domain: {dom_name}",
            f"N det: {len(boxes)}  (gerak={n_gerak} diam={n_diam})",
        ]
        if show_prob:
            info_lines.append(
                f"P  safe={prob[0]:.2f}  warn={prob[1]:.2f}  danger={prob[2]:.2f}"
            )
        for i, line in enumerate(info_lines):
            cv2.putText(out, line, (W - 420, 30 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (220, 220, 220), 1, cv2.LINE_AA)

        return out


# ─────────────────────────────────────────────
# Video runner
# ─────────────────────────────────────────────

def run_video(
    inferencer: DASTRPInferencer,
    src:        str,
    dst:        str  = None,
    log_csv:    str  = None,
    domain_id:  int  = None,
    show_prob:  bool = True,
):
    """
    Proses video frame-by-frame dengan sliding window clip.

    Args:
        inferencer : instance DASTRPInferencer
        src        : path video input
        dst        : path video output (None = tidak disimpan)
        log_csv    : path CSV log per frame (None = tidak disimpan)
        domain_id  : override domain (None = auto dari nama file)
        show_prob  : tampilkan probabilitas di overlay
    """
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Tidak bisa membuka video: {src}")

    fps   = cap.get(cv2.CAP_PROP_FPS)
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Auto-detect domain dari nama file
    if domain_id is None:
        domain_id = infer_domain(Path(src).name)
        d_name    = DOMAIN_NAMES.get(domain_id, "Unknown")
        if domain_id >= 0:
            print(f"Domain auto-detected: {d_name} (dari nama file)")
        else:
            print("Domain tidak terdeteksi — pakai scene classifier")
            domain_id = None

    # Video writer
    writer = None
    if dst:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(dst, fourcc, fps, (fw, fh))
        print(f"Output video : {dst}")

    # CSV writer
    csvf = cw = None
    if log_csv:
        Path(log_csv).parent.mkdir(parents=True, exist_ok=True)
        csvf = open(log_csv, "w", newline="")
        cw   = csv.writer(csvf)
        cw.writerow([
            "frame_id", "risk_id", "risk_label",
            "uncertainty", "n_det", "n_gerak", "n_diam",
            "p_safe", "p_warn", "p_danger",
            "domain_pred", "alpha_safe", "alpha_warn", "alpha_danger",
        ])
        print(f"Log CSV      : {log_csv}")

    # Reset buffer dan tracker
    inferencer.reset()
    print(f"Processing   : {total} frames @ {fps:.2f} FPS")
    print(f"Clip length  : T={inferencer.T}")
    print()

    fid        = 0
    risk_hist  = []   # distribusi risk untuk ringkasan akhir

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = inferencer.predict_frame(frame, domain_id)
            ann    = inferencer.annotate(frame, result, show_prob)

            if writer:
                writer.write(ann)

            if cw:
                p   = result["risk_prob"]
                a   = result["alpha"]
                boxes = result["boxes"]
                mots  = result["motion_pred"]
                n_gerak = int((mots == 0).sum()) if len(mots) > 0 else 0
                n_diam  = int((mots == 1).sum()) if len(mots) > 0 else 0
                cw.writerow([
                    fid,
                    result["risk_id"],
                    RISK_NAMES[result["risk_id"]],
                    f"{result['uncertainty']:.5f}",
                    len(boxes),
                    n_gerak,
                    n_diam,
                    f"{p[0]:.5f}", f"{p[1]:.5f}", f"{p[2]:.5f}",
                    DOMAIN_NAMES.get(result["domain_pred"], "?"),
                    f"{a[0]:.3f}", f"{a[1]:.3f}", f"{a[2]:.3f}",
                ])

            risk_hist.append(result["risk_id"])

            if fid % 200 == 0 or fid == total - 1:
                print(f"  frame {fid:5d}/{total}  |  "
                      f"risk={RISK_NAMES[result['risk_id']]:7s}  |  "
                      f"unc={result['uncertainty']:.3f}  |  "
                      f"n_det={len(result['boxes'])}")
            fid += 1

    finally:
        cap.release()
        if writer:
            writer.release()
        if csvf:
            csvf.close()

    # ── Ringkasan ─────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Selesai — {fid} frame diproses")
    if risk_hist:
        from collections import Counter
        dist = Counter(risk_hist)
        for r_id, r_name in RISK_NAMES.items():
            n   = dist.get(r_id, 0)
            pct = n / fid * 100
            print(f"  {r_name:8s}: {n:5d} frame ({pct:5.1f}%)")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DASTRP-Net inference pada video"
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path ke best_model.pth")
    parser.add_argument("--yolo",       required=True,
                        help="Path ke best.pt YOLO (1 kelas motor)")
    parser.add_argument("--input",      required=True,
                        help="Path video input")
    parser.add_argument("--output",     default=None,
                        help="Path video output teranotasi")
    parser.add_argument("--log-csv",    default=None,
                        help="Path CSV log per frame")
    parser.add_argument("--T",          type=int, default=4,
                        help="Clip length — harus sama dengan saat training")
    parser.add_argument("--img-size",   type=int, default=640)
    parser.add_argument("--domain",     type=int, default=None,
                        help="0=UGM 1=Utara 2=Ringroad "
                             "(None=auto dari nama file/scene clf)")
    parser.add_argument("--no-prob",    action="store_true",
                        help="Sembunyikan probabilitas di overlay")
    parser.add_argument("--device",     default=None,
                        help="'cuda' atau 'cpu' (default: auto)")
    add_seed_args(parser)
    args = parser.parse_args()
    set_seed(args.seed, args.deterministic)

    inf = DASTRPInferencer(
        checkpoint   = args.checkpoint,
        yolo_weights = args.yolo,
        T            = args.T,
        img_size     = args.img_size,
        device       = args.device,
    )

    run_video(
        inferencer = inf,
        src        = args.input,
        dst        = args.output,
        log_csv    = args.log_csv,
        domain_id  = args.domain,
        show_prob  = not args.no_prob,
    )
