"""
models/baseline_b1.py
=====================
B1 Baseline: YOLO11n + MLP (Frame-Only, No Temporal Modeling)

Digunakan sebagai baseline ablasi untuk mengukur kontribusi
temporal modeling dalam DASTRP-Net.

Pipeline:
  Input : (B, T, 3, H, W) clip — sama dengan DASTRPNet
  ──────────────────────────────────────────────────────
  Per frame t:
    1. YOLO11n backbone (frozen) → GAP embedding (B, 256)
  ──────────────────────────────────────────────────────
  Setelah semua T frame:
    2. Mean pool T frame embeddings → clip embedding (B, 256)
    3. 3-layer MLP → risk logits (B, 3)
    4. Softmax → risk prediction + pseudo-uncertainty

Perbedaan fundamental dari DASTRPNet:
  - TIDAK ada GNN   → tidak ada interaksi antar motor
  - TIDAK ada CBN   → tidak ada domain adaptation
  - TIDAK ada CrossModalFusion → frame emb langsung ke head
  - TIDAK ada VelocityTracker → tidak ada motion context
  - TIDAK ada EDL   → softmax biasa, bukan Dirichlet
  - TIDAK ada ObjectRiskHead → tidak ada per-objek prediction

Output dict kompatibel dengan train.py dan evaluate.py:
  - Semua key yang diperlukan tersedia
  - Key yang tidak relevan (domain_logits, obj_risk_*) 
    diisi dengan dummy tensor yang tidak ikut loss

Cara training: gunakan train.py dengan --model b1
  python scripts/train.py --model b1 --yolo ... --dataset ...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Import YOLODetector dari architecture.py yang sudah ada
# agar backbone behavior identik dengan DASTRPNet
import sys
sys.path.insert(0, str(Path(__file__).parent))
from architecture import YOLODetector

NUM_RISK    = 3
NUM_DOMAINS = 3
FRAME_DIM   = 256


# ─────────────────────────────────────────────
# MLP Risk Head (standard softmax, bukan EDL)
# ─────────────────────────────────────────────

class MLPRiskHead(nn.Module):
    """
    3-layer MLP dengan softmax untuk klasifikasi risk.
    Identik strukturnya dengan EvidentialRiskHead tapi
    output-nya probabilitas softmax, bukan Dirichlet.
    """

    def __init__(self, in_dim: int = FRAME_DIM, num_classes: int = NUM_RISK):
        super().__init__()
        self.K   = num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),    nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        x : (B, in_dim)

        Returns dict dengan key sama persis seperti EvidentialRiskHead
        agar compute_loss() dan evaluate() tidak perlu diubah.
        """
        logits    = self.net(x)                          # (B, K)
        risk_prob = F.softmax(logits, dim=1)             # (B, K)

        # uncertainty = entropi ternormalisasi sebagai proxy
        # (bukan vacuity EDL, tapi konsisten secara range [0,1])
        log_K  = torch.log(torch.tensor(float(self.K), device=x.device))
        entropy = -(risk_prob * (risk_prob + 1e-8).log()).sum(1)
        unc     = (entropy / log_K).clamp(0, 1)          # (B,)

        return {
            "alpha":       risk_prob + 1.0,   # dummy alpha agar compute_loss tidak crash
            "risk_prob":   risk_prob,
            "risk_pred":   risk_prob.argmax(1),
            "uncertainty": unc,
            "logits":      logits,            # untuk CE loss langsung
        }

    @staticmethod
    def edl_loss(alpha, labels, epoch, num_epochs, lam=1.0):
        """
        Override EDL loss → standard cross-entropy.
        alpha di sini adalah (risk_prob + 1) dari forward(),
        jadi risk_prob = alpha - 1.
        """
        risk_prob = alpha - 1.0
        logits    = (risk_prob + 1e-8).log()   # log-softmax sebagai logit proxy
        return F.cross_entropy(logits, labels)


# ─────────────────────────────────────────────
# B1 Model: YOLO11n + MLP
# ─────────────────────────────────────────────

class BaselineB1(nn.Module):
    """
    B1 Baseline — YOLO11n + MLP Risk Classifier (Frame-Only).

    Tidak ada temporal modeling, tidak ada GNN, tidak ada domain adaptation.
    Setiap frame diproses secara independen; prediksi final adalah mean
    pooling embedding T frame → MLP.

    Interface identik dengan DASTRPNet sehingga train.py dan evaluate.py
    dapat digunakan tanpa modifikasi.

    Args:
        yolo_weights : path ke best.pt YOLO11n
        yolo_conf    : confidence threshold (hanya dipakai saat inference)
        yolo_iou     : NMS IoU threshold
        T_agg        : None = rata-rata semua T frame (sama seperti DASTRPNet)
    """

    def __init__(
        self,
        yolo_weights: str,
        yolo_conf:    float = 0.25,
        yolo_iou:     float = 0.45,
        T_agg:        int   = None,
        **kwargs,   # terima argumen DASTRPNet yang tidak dipakai (dist_thresh, ablation)
    ):
        super().__init__()
        self.T_agg = T_agg

        # ── Backbone identik dengan DASTRPNet ─────────────────────
        self.yolo = YOLODetector(yolo_weights, yolo_conf, yolo_iou)

        # ── Frame projection (sama dengan DASTRPNet) ───────────────
        self.frame_proj = nn.Sequential(
            nn.Linear(256, FRAME_DIM), nn.ReLU()
        )

        # ── MLP Risk Head (menggantikan seluruh GNN+Fusion+EDL) ────
        self.risk_head = MLPRiskHead(in_dim=FRAME_DIM)

        # ── Dummy domain classifier ────────────────────────────────
        # Diperlukan agar compute_loss() tidak crash saat mengakses
        # outputs["domain_logits"] — nilainya tidak ikut loss karena
        # lambda_domain dapat di-set ke 0 di konfigurasi B1
        self.domain_clf = nn.Linear(FRAME_DIM, NUM_DOMAINS)

    # ── Public API (sama dengan DASTRPNet) ─────────────────────────

    def reset_tracker(self):
        """No-op — B1 tidak punya velocity tracker."""
        pass

    def forward(
        self,
        clip_images:    torch.Tensor,         # (B, T, 3, H, W)
        domain_ids:     torch.Tensor = None,  # tidak dipakai tapi diterima
        gt_boxes_seq:   list         = None,  # tidak dipakai tapi diterima
        epoch:          int          = 0,
        num_epochs:     int          = 100,
    ) -> dict:
        """
        Forward pass B1.

        Alur per frame:
          images_t → YOLO backbone → GAP → frame_proj → frame_emb_t

        Alur clip:
          mean(frame_emb_0 ... frame_emb_{T-1}) → MLP → risk_pred

        Output dict kompatibel dengan train.py dan evaluate.py:
          - risk_pred, risk_prob, uncertainty : sama seperti DASTRPNet
          - all_frame_risks : (B, T, 3) — diisi dengan prediksi per frame
          - domain_logits   : (B, 3)    — dari domain_clf dummy
          - det_boxes       : list[B] tensor(0, 4) — kosong (tidak ada detection)
          - motion_logits   : list[B] tensor(0, 2) — kosong
          - obj_risk_logits : list[B] tensor(0, 3) — kosong
          - obj_risk_pred   : list[B] tensor(0,)   — kosong
        """
        B, T   = clip_images.shape[:2]
        device = clip_images.device

        all_frame_embs  = []   # (B, FRAME_DIM) per frame
        all_frame_probs = []   # (B, 3) per frame

        for t in range(T):
            images_t = clip_images[:, t]   # (B, 3, H, W)

            # Step 1 — YOLO backbone (frozen, fp32)
            _, frame_emb_raw = self.yolo.extract_features(images_t)

            # Step 2 — projection ke FRAME_DIM
            frame_emb = self.frame_proj(frame_emb_raw.to(images_t.dtype))

            # Step 3 — per-frame risk prediction (untuk all_frame_risks)
            with torch.no_grad() if t < T - 1 else torch.enable_grad():
                risk_t = self.risk_head(frame_emb)
            all_frame_probs.append(risk_t["risk_prob"].detach())

            all_frame_embs.append(frame_emb)

        # ── Temporal aggregation (mean pool T frame embeddings) ────
        frame_seq = torch.stack(all_frame_embs, dim=1)   # (B, T, FRAME_DIM)
        if self.T_agg is not None:
            agg = frame_seq[:, -self.T_agg:].mean(dim=1)
        else:
            agg = frame_seq.mean(dim=1)                  # (B, FRAME_DIM)

        # ── Risk prediction final ──────────────────────────────────
        risk_out = self.risk_head(agg)

        # ── Domain logits (dummy) ─────────────────────────────────
        domain_logits = self.domain_clf(agg.detach())    # (B, 3)

        # ── all_frame_risks ────────────────────────────────────────
        all_frame_risks = torch.stack(all_frame_probs, dim=1)  # (B, T, 3)

        # ── Dummy per-objek output ─────────────────────────────────
        # B1 tidak melakukan detection sehingga N=0 untuk semua sample
        empty_boxes = [torch.zeros(0, 4,  device=device) for _ in range(B)]
        empty_mot   = [torch.zeros(0, 2,  device=device) for _ in range(B)]
        empty_orisk = [torch.zeros(0, 3,  device=device) for _ in range(B)]
        empty_opred = [torch.zeros(0,     device=device,
                                   dtype=torch.long)      for _ in range(B)]

        return {
            # Output utama
            "risk_pred":       risk_out["risk_pred"],      # (B,)
            "risk_prob":       risk_out["risk_prob"],      # (B, 3)
            "uncertainty":     risk_out["uncertainty"],    # (B,)
            "alpha":           risk_out["alpha"],          # (B, 3) dummy

            # Untuk loss
            "domain_logits":   domain_logits,             # (B, 3)
            "all_frame_risks": all_frame_risks,           # (B, T, 3)

            # Per-objek (kosong — B1 tidak melakukan detection)
            "det_boxes":       empty_boxes,
            "motion_logits":   empty_mot,
            "obj_risk_logits": empty_orisk,
            "obj_risk_pred":   empty_opred,
        }
