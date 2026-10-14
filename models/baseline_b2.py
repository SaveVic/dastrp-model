"""
models/baseline_b2.py
=====================
B2 Baseline: YOLO11n + Bidirectional LSTM (Temporal, No Graph)

Digunakan sebagai baseline untuk mengukur kontribusi
graph-based interaction modeling dalam DASTRP-Net.
B2 menambahkan temporal modeling (BiLSTM) di atas B1,
tapi tanpa interaksi spasial antar objek.

Pipeline:
  Input : (B, T, 3, H, W) clip — sama dengan DASTRPNet
  ──────────────────────────────────────────────────────
  Per frame t:
    1. YOLO11n backbone (frozen) → GAP embedding (B, 256)
    2. Linear projection → frame_emb_t (B, FRAME_DIM)
  ──────────────────────────────────────────────────────
  Sequence modeling:
    3. Stack T frame embeddings → (B, T, FRAME_DIM)
    4. BiLSTM → hidden state per frame (B, T, 2*LSTM_DIM)
    5. Context vector = concat [forward h_T, backward h_0]
                      → (B, 2*LSTM_DIM)
    6. Linear projection → clip embedding (B, FUSION_DIM)
    7. MLP Risk Head → risk_pred + pseudo-uncertainty

Perbedaan dari DASTRPNet:
  ✓ Ada temporal modeling (BiLSTM menggantikan GNN+Fusion)
  ✗ Tidak ada GNN   → tidak ada interaksi spasial antar motor
  ✗ Tidak ada CBN   → tidak ada domain adaptation
  ✗ Tidak ada velocity tracker → tidak dipakai oleh BiLSTM
  ✗ Tidak ada EDL   → softmax biasa
  ✗ Tidak ada ObjectRiskHead

Perbedaan dari B1:
  ✓ B2 memiliki temporal modeling eksplisit (BiLSTM)
  ✗ B1 hanya melakukan mean pooling frame embeddings

Output dict: kompatibel penuh dengan train.py dan evaluate.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from architecture import YOLODetector
from baseline_b1 import MLPRiskHead   # reuse — sama persis

NUM_RISK    = 3
NUM_DOMAINS = 3
FRAME_DIM   = 256   # frame embedding dim (sama dengan DASTRPNet)
LSTM_DIM    = 256   # hidden dim per direction BiLSTM
FUSION_DIM  = 256   # output dim setelah projection BiLSTM


# ─────────────────────────────────────────────
# BiLSTM Temporal Encoder
# ─────────────────────────────────────────────

class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM untuk encoding sekuens T frame embeddings.

    Input  : (B, T, FRAME_DIM)  — sequence frame embeddings
    Output : (B, FUSION_DIM)    — clip-level representation

    Cara kerja:
      - Forward LSTM memproses frame t=0 → T-1
      - Backward LSTM memproses frame t=T-1 → 0
      - Context vector = concat [h_T_forward, h_0_backward]
        → menggabungkan informasi awal dan akhir clip
      - Linear projection → FUSION_DIM

    Ini berbeda dari DASTRPNet yang menggunakan CrossModalFusion
    (scene context + motion graph via cross-attention).
    BiLSTM hanya melihat urutan temporal frame embeddings
    tanpa interaksi antar objek di dalam frame yang sama.

    Args:
        input_dim  : dimensi input per frame (= FRAME_DIM)
        hidden_dim : dimensi hidden state per direction
        num_layers : jumlah layer BiLSTM (default: 2)
        dropout    : dropout antar layer (hanya aktif jika num_layers > 1)
        out_dim    : dimensi output setelah projection
    """

    def __init__(
        self,
        input_dim:  int   = FRAME_DIM,
        hidden_dim: int   = LSTM_DIM,
        num_layers: int   = 2,
        dropout:    float = 0.3,
        out_dim:    int   = FUSION_DIM,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # BiLSTM: output dim = 2 * hidden_dim (forward + backward)
        self.lstm = nn.LSTM(
            input_size    = input_dim,
            hidden_size   = hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            bidirectional = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )

        # Layer norm untuk stabilitas
        self.norm = nn.LayerNorm(2 * hidden_dim)

        # Projection ke FUSION_DIM
        self.proj = nn.Sequential(
            nn.Linear(2 * hidden_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        x : (B, T, input_dim)

        Returns:
            clip_emb    : (B, out_dim)   — clip-level representation
            hidden_seq  : (B, T, 2*hidden_dim) — per-frame hidden states
        """
        # LSTM forward
        # output: (B, T, 2*hidden_dim) — semua hidden states
        # h_n  : (num_layers*2, B, hidden_dim) — final hidden state
        output, (h_n, _) = self.lstm(x)

        # Context vector: concat final forward + final backward hidden state
        # h_n shape: (num_layers*2, B, hidden_dim)
        # Layer terakhir: forward = h_n[-2], backward = h_n[-1]
        h_forward  = h_n[-2]   # (B, hidden_dim) — dari frame T-1
        h_backward = h_n[-1]   # (B, hidden_dim) — dari frame 0
        context    = torch.cat([h_forward, h_backward], dim=1)  # (B, 2*hidden_dim)

        # Normalisasi + projection
        context  = self.norm(context)
        clip_emb = self.proj(context)    # (B, out_dim)

        return clip_emb, output          # output untuk per-frame risks


# ─────────────────────────────────────────────
# B2 Model: YOLO11n + BiLSTM
# ─────────────────────────────────────────────

class BaselineB2(nn.Module):
    """
    B2 Baseline — YOLO11n + BiLSTM Risk Classifier.

    Menambahkan temporal modeling eksplisit via Bidirectional LSTM
    di atas B1 (frame-only MLP), tapi tanpa interaksi spasial
    antar objek (tidak ada GNN) dan tanpa domain adaptation (tidak ada CBN).

    Perbandingan dengan DASTRPNet:
      B1  → mean pool frames → MLP        (no temporal)
      B2  → BiLSTM sequence → MLP         (temporal, no graph)
      Exp3→ GATv2 graph + cross-attention  (temporal + graph + domain)

    Interface identik dengan DASTRPNet sehingga train.py dan evaluate.py
    dapat digunakan tanpa modifikasi.

    Args:
        yolo_weights : path ke best.pt YOLO11n
        yolo_conf    : confidence threshold
        yolo_iou     : NMS IoU threshold
        lstm_dim     : hidden dim BiLSTM per direction (default: 256)
        lstm_layers  : jumlah layer BiLSTM (default: 2)
        lstm_dropout : dropout antar LSTM layer
        T_agg        : tidak dipakai di B2 (BiLSTM sudah menangani seluruh T)
    """

    def __init__(
        self,
        yolo_weights:  str,
        yolo_conf:     float = 0.25,
        yolo_iou:      float = 0.45,
        lstm_dim:      int   = LSTM_DIM,
        lstm_layers:   int   = 2,
        lstm_dropout:  float = 0.3,
        T_agg:         int   = None,    # diterima tapi tidak dipakai
        **kwargs,
    ):
        super().__init__()

        # ── Backbone identik dengan DASTRPNet ─────────────────────
        self.yolo = YOLODetector(yolo_weights, yolo_conf, yolo_iou)

        # ── Frame projection (sama dengan DASTRPNet) ───────────────
        self.frame_proj = nn.Sequential(
            nn.Linear(256, FRAME_DIM), nn.ReLU()
        )

        # ── BiLSTM Temporal Encoder ────────────────────────────────
        self.lstm_encoder = BiLSTMEncoder(
            input_dim  = FRAME_DIM,
            hidden_dim = lstm_dim,
            num_layers = lstm_layers,
            dropout    = lstm_dropout,
            out_dim    = FUSION_DIM,
        )

        # ── MLP Risk Head (softmax, sama dengan B1) ────────────────
        self.risk_head = MLPRiskHead(in_dim=FUSION_DIM)

        # ── Per-frame risk head (untuk all_frame_risks loss) ───────
        # Input: hidden state BiLSTM per frame (2*lstm_dim)
        self.frame_risk_head = nn.Sequential(
            nn.Linear(2 * lstm_dim, 64), nn.ReLU(),
            nn.Linear(64, NUM_RISK),
        )

        # ── Dummy domain classifier ────────────────────────────────
        self.domain_clf = nn.Linear(FUSION_DIM, NUM_DOMAINS)

        print(f"[BaselineB2] BiLSTM: dim={lstm_dim}×2, "
              f"layers={lstm_layers}, dropout={lstm_dropout}")

    def reset_tracker(self):
        """No-op — B2 tidak punya velocity tracker."""
        pass

    def forward(
        self,
        clip_images:    torch.Tensor,
        domain_ids:     torch.Tensor = None,
        gt_boxes_seq:   list         = None,
        epoch:          int          = 0,
        num_epochs:     int          = 100,
    ) -> dict:
        """
        Forward pass B2.

        Alur:
          images_t → YOLO backbone → frame_proj → frame_emb_t
          [frame_emb_0, ..., frame_emb_{T-1}] → BiLSTM → clip_emb
          clip_emb → MLP Risk Head → risk_pred

        Per-frame risks diperoleh dari hidden states BiLSTM
        (bukan prediksi independen seperti B1).
        """
        B, T   = clip_images.shape[:2]
        device = clip_images.device

        # ── Step 1: Ekstrak frame embeddings (per frame) ───────────
        frame_embs = []
        for t in range(T):
            images_t = clip_images[:, t]   # (B, 3, H, W)
            _, frame_emb_raw = self.yolo.extract_features(images_t)
            frame_emb        = self.frame_proj(
                frame_emb_raw.to(images_t.dtype)
            )                              # (B, FRAME_DIM)
            frame_embs.append(frame_emb)

        # ── Step 2: BiLSTM temporal encoding ──────────────────────
        # Stack frame embeddings: (B, T, FRAME_DIM)
        frame_seq = torch.stack(frame_embs, dim=1)

        # clip_emb  : (B, FUSION_DIM)
        # hidden_seq: (B, T, 2*LSTM_DIM)
        clip_emb, hidden_seq = self.lstm_encoder(frame_seq)

        # ── Step 3: Risk prediction dari clip embedding ─────────────
        risk_out = self.risk_head(clip_emb)

        # ── Step 4: Per-frame risk dari BiLSTM hidden states ───────
        # Ini lebih informatif dari prediksi frame independen (B1)
        # karena hidden_t sudah mengandung konteks dari frame sebelumnya
        frame_logits = self.frame_risk_head(
            hidden_seq.detach()
        )                                             # (B, T, NUM_RISK)
        all_frame_probs = F.softmax(frame_logits, dim=2)  # (B, T, 3)

        # ── Step 5: Domain logits (dummy) ─────────────────────────
        domain_logits = self.domain_clf(clip_emb.detach())  # (B, 3)

        # ── Step 6: Dummy per-objek output ────────────────────────
        empty_boxes = [torch.zeros(0, 4, device=device) for _ in range(B)]
        empty_mot   = [torch.zeros(0, 2, device=device) for _ in range(B)]
        empty_orisk = [torch.zeros(0, 3, device=device) for _ in range(B)]
        empty_opred = [torch.zeros(0,    device=device,
                                   dtype=torch.long)    for _ in range(B)]

        return {
            # Output utama
            "risk_pred":       risk_out["risk_pred"],
            "risk_prob":       risk_out["risk_prob"],
            "uncertainty":     risk_out["uncertainty"],
            "alpha":           risk_out["alpha"],

            # Untuk loss
            "domain_logits":   domain_logits,
            "all_frame_risks": all_frame_probs,    # (B, T, 3) dari BiLSTM hidden

            # Per-objek (kosong)
            "det_boxes":       empty_boxes,
            "motion_logits":   empty_mot,
            "obj_risk_logits": empty_orisk,
            "obj_risk_pred":   empty_opred,
        }
