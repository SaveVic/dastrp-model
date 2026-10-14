"""
models/baseline_b3.py
=====================
B3 Baseline: YOLO11n + Transformer Encoder (Temporal, No Graph)

Digunakan sebagai baseline untuk membandingkan attention-based
temporal modeling dengan recurrent (B2-BiLSTM) dan graph-based
(DASTRPNet-GATv2) temporal modeling.

Pipeline:
  Input : (B, T, 3, H, W) clip — sama dengan DASTRPNet
  ──────────────────────────────────────────────────────
  Per frame t:
    1. YOLO11n backbone (frozen) → GAP embedding (B, 256)
    2. Linear projection → frame_emb_t (B, FRAME_DIM)
  ──────────────────────────────────────────────────────
  Sequence modeling:
    3. Stack T frame embeddings → (B, T, FRAME_DIM)
    4. Positional encoding (sinusoidal) → (B, T, FRAME_DIM)
    5. Transformer Encoder (N layers, H heads) → (B, T, FRAME_DIM)
    6. [CLS] token aggregation → (B, FRAME_DIM)
       atau mean pool semua T tokens → (B, FRAME_DIM)
    7. MLP Risk Head → risk_pred + pseudo-uncertainty

Perbedaan dari DASTRPNet:
  ✓ Ada temporal modeling (Transformer Encoder)
  ✗ Tidak ada GNN   → tidak ada interaksi spasial antar motor
  ✗ Tidak ada CBN   → tidak ada domain adaptation
  ✗ Tidak ada velocity tracker
  ✗ Tidak ada EDL   → softmax biasa
  ✗ Tidak ada ObjectRiskHead

Perbandingan antar baseline:
  B1  → mean pool T frames     (no temporal)
  B2  → BiLSTM sequence        (recurrent temporal)
  B3  → Transformer Encoder    (attention-based temporal)  ← ini
  Exp → GATv2 + cross-attention (graph + domain adaptive)

Mengapa Transformer dibanding BiLSTM:
  - BiLSTM (B2): memproses sekuensial, konteks bergantung urutan
  - Transformer (B3): semua frame attend ke semua frame secara parallel,
    tidak ada bias posisional — lebih fleksibel tapi butuh posisi eksplisit
  - Dengan T=4, Transformer dapat melihat semua 4 frame sekaligus
    dan belajar frame mana yang paling relevan untuk prediksi risk

Output dict: kompatibel penuh dengan train.py dan evaluate.py.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from architecture import YOLODetector
from baseline_b1 import MLPRiskHead

NUM_RISK    = 3
NUM_DOMAINS = 3
FRAME_DIM   = 256
FUSION_DIM  = 256


# ─────────────────────────────────────────────
# Sinusoidal Positional Encoding
# ─────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """
    Positional encoding sinusoidal standar (Vaswani et al. 2017).

    Penting untuk Transformer karena self-attention bersifat
    permutation-invariant — tanpa PE, urutan frame t=0,1,2,3
    tidak dapat dibedakan dari urutan acak.

    Dengan T=4 frame, PE memberikan informasi temporal:
      t=0 → vektor posisi awal clip
      t=3 → vektor posisi akhir clip

    PE ditambahkan ke frame embedding (bukan concatenate)
    agar dimensi tetap FRAME_DIM.
    """

    def __init__(self, d_model: int = FRAME_DIM, max_len: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Precompute sinusoidal matrix
        pe      = torch.zeros(max_len, d_model)   # (max_len, d_model)
        pos     = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)                       # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, d_model)
        Returns: (B, T, d_model) — x + PE[:T]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ─────────────────────────────────────────────
# Transformer Temporal Encoder
# ─────────────────────────────────────────────

class TransformerTemporalEncoder(nn.Module):
    """
    Transformer Encoder untuk encoding sekuens T frame embeddings.

    Arsitektur:
      1. [CLS] token prepend → (B, T+1, FRAME_DIM)
      2. Positional encoding sinusoidal
      3. N × TransformerEncoderLayer (multi-head self-attention + FFN)
      4. [CLS] token output → clip representation (B, FRAME_DIM)

    [CLS] token (learnable) adalah idiom standar BERT/ViT:
      - Di-prepend ke sekuens frame embeddings
      - Di setiap layer, CLS attend ke semua frame
      - Output CLS di layer terakhir = representasi global clip
      - Alternatif: mean pool semua output tokens (dikendalikan
        oleh parameter `pool_mode`)

    Args:
        d_model    : dimensi embedding (= FRAME_DIM)
        n_heads    : jumlah attention heads
        n_layers   : jumlah Transformer encoder layers
        dim_ff     : dimensi FFN inner layer (biasanya 4×d_model)
        dropout    : dropout di attention dan FFN
        pool_mode  : 'cls' (CLS token) atau 'mean' (mean all tokens)
        out_dim    : dimensi output setelah projection
    """

    def __init__(
        self,
        d_model:   int   = FRAME_DIM,
        n_heads:   int   = 4,
        n_layers:  int   = 2,
        dim_ff:    int   = 512,
        dropout:   float = 0.1,
        pool_mode: str   = "cls",
        out_dim:   int   = FUSION_DIM,
    ):
        super().__init__()
        assert pool_mode in ("cls", "mean"), \
            "pool_mode harus 'cls' atau 'mean'"
        self.pool_mode = pool_mode
        self.d_model   = d_model

        # [CLS] token (learnable) — hanya dipakai jika pool_mode='cls'
        if pool_mode == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(
            d_model=d_model, dropout=dropout
        )

        # Transformer Encoder layers
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = n_heads,
            dim_feedforward = dim_ff,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,   # (B, T, d_model) — tidak perlu transpose
            norm_first      = True,   # Pre-LN: lebih stabil dari Post-LN
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer = enc_layer,
            num_layers    = n_layers,
            norm          = nn.LayerNorm(d_model),
        )

        # Output projection → out_dim
        self.proj = nn.Sequential(
            nn.Linear(d_model, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """
        x : (B, T, d_model) — sekuens frame embeddings

        Returns:
            clip_emb   : (B, out_dim)   — clip representation
            token_out  : (B, T, d_model) — per-token output (untuk frame risks)
        """
        B, T, _ = x.shape

        if self.pool_mode == "cls":
            # Prepend [CLS] token: (B, T+1, d_model)
            cls = self.cls_token.expand(B, -1, -1)
            x   = torch.cat([cls, x], dim=1)

        # Positional encoding
        x = self.pos_enc(x)

        # Transformer encoder
        out = self.transformer(x)   # (B, T atau T+1, d_model)

        if self.pool_mode == "cls":
            clip_repr  = out[:, 0]    # (B, d_model) — [CLS] token
            token_out  = out[:, 1:]   # (B, T, d_model) — frame tokens
        else:
            clip_repr  = out.mean(dim=1)   # (B, d_model) — mean pool
            token_out  = out               # (B, T, d_model)

        clip_emb = self.proj(clip_repr)    # (B, out_dim)
        return clip_emb, token_out


# ─────────────────────────────────────────────
# B3 Model: YOLO11n + Transformer Encoder
# ─────────────────────────────────────────────

class BaselineB3(nn.Module):
    """
    B3 Baseline — YOLO11n + Transformer Encoder.

    Menggunakan multi-head self-attention untuk temporal modeling,
    sebagai alternatif recurrent (B2-BiLSTM) dan graph-based
    (DASTRPNet-GATv2) approaches.

    Relevansi untuk perbandingan:
      B1  → tidak ada temporal → akurasi paling rendah diharapkan
      B2  → BiLSTM (recurrent) → temporal + sequential inductive bias
      B3  → Transformer (attention) → temporal + global context
      Exp → GATv2 + domain adapt → graph + temporal + domain

    Interface identik dengan DASTRPNet sehingga train.py dan evaluate.py
    dapat digunakan tanpa modifikasi.

    Args:
        yolo_weights   : path ke best.pt YOLO11n
        yolo_conf      : confidence threshold
        yolo_iou       : NMS IoU threshold
        tf_n_heads     : jumlah attention heads (default: 4)
        tf_n_layers    : jumlah Transformer layers (default: 2)
        tf_dim_ff      : FFN inner dim (default: 512 = 2×FRAME_DIM)
        tf_dropout     : dropout (default: 0.1)
        tf_pool_mode   : 'cls' atau 'mean' (default: 'cls')
        T_agg          : tidak dipakai — Transformer menangani seluruh T
    """

    def __init__(
        self,
        yolo_weights:  str,
        yolo_conf:     float = 0.25,
        yolo_iou:      float = 0.45,
        tf_n_heads:    int   = 4,
        tf_n_layers:   int   = 2,
        tf_dim_ff:     int   = 512,
        tf_dropout:    float = 0.1,
        tf_pool_mode:  str   = "cls",
        T_agg:         int   = None,
        **kwargs,
    ):
        super().__init__()
        self.tf_pool_mode = tf_pool_mode

        # ── Backbone identik dengan DASTRPNet ─────────────────────
        self.yolo = YOLODetector(yolo_weights, yolo_conf, yolo_iou)

        # ── Frame projection (sama dengan DASTRPNet) ───────────────
        self.frame_proj = nn.Sequential(
            nn.Linear(256, FRAME_DIM), nn.ReLU()
        )

        # ── Transformer Temporal Encoder ──────────────────────────
        self.transformer = TransformerTemporalEncoder(
            d_model   = FRAME_DIM,
            n_heads   = tf_n_heads,
            n_layers  = tf_n_layers,
            dim_ff    = tf_dim_ff,
            dropout   = tf_dropout,
            pool_mode = tf_pool_mode,
            out_dim   = FUSION_DIM,
        )

        # ── MLP Risk Head (softmax, sama dengan B1/B2) ─────────────
        self.risk_head = MLPRiskHead(in_dim=FUSION_DIM)

        # ── Per-frame risk head (dari Transformer token outputs) ───
        # Menggunakan token output per frame untuk supervisi
        # all_frame_risks — lebih informatif dari prediksi independen
        self.frame_risk_head = nn.Sequential(
            nn.Linear(FRAME_DIM, 64), nn.ReLU(),
            nn.Linear(64, NUM_RISK),
        )

        # ── Dummy domain classifier ────────────────────────────────
        self.domain_clf = nn.Linear(FUSION_DIM, NUM_DOMAINS)

        print(f"[BaselineB3] Transformer: heads={tf_n_heads}, "
              f"layers={tf_n_layers}, ff={tf_dim_ff}, "
              f"pool={tf_pool_mode}")
        n_tf = sum(p.numel() for p in self.transformer.parameters())
        print(f"  Transformer params: {n_tf:,}")

    def reset_tracker(self):
        """No-op — B3 tidak punya velocity tracker."""
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
        Forward pass B3.

        Alur:
          images_t → YOLO backbone → frame_proj → frame_emb_t
          [frame_emb_0, ..., frame_emb_{T-1}] + PE → Transformer
          → [CLS] output → MLP Risk Head → risk_pred

        Per-frame risks diperoleh dari Transformer token outputs
        (bukan prediksi independen seperti B1).
        """
        B, T   = clip_images.shape[:2]
        device = clip_images.device

        # ── Step 1: Ekstrak frame embeddings ──────────────────────
        frame_embs = []
        for t in range(T):
            images_t = clip_images[:, t]   # (B, 3, H, W)
            _, frame_emb_raw = self.yolo.extract_features(images_t)
            frame_emb        = self.frame_proj(
                frame_emb_raw.to(images_t.dtype)
            )                              # (B, FRAME_DIM)
            frame_embs.append(frame_emb)

        # ── Step 2: Transformer temporal encoding ─────────────────
        # Stack: (B, T, FRAME_DIM)
        frame_seq = torch.stack(frame_embs, dim=1)

        # clip_emb   : (B, FUSION_DIM)
        # token_out  : (B, T, FRAME_DIM) — Transformer token output
        clip_emb, token_out = self.transformer(frame_seq)

        # ── Step 3: Risk prediction dari clip embedding ─────────────
        risk_out = self.risk_head(clip_emb)

        # ── Step 4: Per-frame risks dari Transformer token outputs ──
        # token_out[:, t, :] sudah mengandung global context via attention
        frame_logits    = self.frame_risk_head(
            token_out.detach()
        )                                        # (B, T, NUM_RISK)
        all_frame_probs = F.softmax(frame_logits, dim=2)  # (B, T, 3)

        # ── Step 5: Domain logits (dummy) ─────────────────────────
        domain_logits = self.domain_clf(clip_emb.detach())   # (B, 3)

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
            "all_frame_risks": all_frame_probs,   # (B, T, 3) dari token outputs

            # Per-objek (kosong)
            "det_boxes":       empty_boxes,
            "motion_logits":   empty_mot,
            "obj_risk_logits": empty_orisk,
            "obj_risk_pred":   empty_opred,
        }
