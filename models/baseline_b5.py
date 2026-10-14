"""
models/baseline_b5.py
=====================
B5 Baseline: SlowFast Network (Feichtenhofer et al., ICCV 2019)
Adapted untuk three-class motorcycle risk classification.

Re-implementasi dari scratch berdasarkan:
  Feichtenhofer, C. et al., "SlowFast Networks for Video Recognition,"
  Proc. IEEE/CVF ICCV, 2019, pp. 6202-6211.

Adaptasi untuk dataset ini:
  - Input: clip (B, T, 3, H, W) dari DASTRPClipDataset (T=4)
  - Slow pathway: T_slow = T//alpha frame (alpha=4 → T_slow=1)
  - Fast pathway: seluruh T=4 frame
  - Task: 3-class risk (Safe/Warning/Danger)
  - Detection-free: raw pixel input, tanpa YOLO

Kunci perbaikan vs versi sebelumnya:
  - lateral_channels dihitung secara dinamis di FastPathway.__init__
    dan dipass sebagai argumen ke SlowPathway.__init__
  - SlowPathway tidak menghitung channel sendiri — channel diterima
    dari FastPathway agar selalu konsisten
  - lat_fuse per stage menerima fuse_in = slow_ch + lat_ch yang
    dihitung iteratif (bukan hardcode)

Output dict: kompatibel penuh dengan train.py dan evaluate.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from baseline_b1 import MLPRiskHead

NUM_RISK    = 3
NUM_DOMAINS = 3
ALPHA       = 4
BETA_INV    = 8
C_SLOW_BASE = 64
C_FAST_BASE = C_SLOW_BASE // BETA_INV   # 8


# ─────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────

class Conv3dBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel,
                 stride=(1,1,1), padding=(0,0,0)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class ResBlock3D(nn.Module):
    """3D Residual Block dengan optional spatial downsampling."""
    def __init__(self, in_ch, out_ch, stride_hw=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, (3,3,3),
                               stride=(1,stride_hw,stride_hw),
                               padding=(1,1,1), bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, (1,3,3),
                               padding=(0,1,1), bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)
        self.skip  = None
        if in_ch != out_ch or stride_hw != 1:
            self.skip = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1,
                          stride=(1,stride_hw,stride_hw), bias=False),
                nn.BatchNorm3d(out_ch),
            )

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(h + (self.skip(x) if self.skip else x))


# ─────────────────────────────────────────────
# Fast Pathway
# ─────────────────────────────────────────────

class FastPathway(nn.Module):
    """
    High temporal resolution, low channel count.
    Memproses semua T frame.

    Channel plan (C_FAST_BASE=8):
      stem    : 3  → 8
      stage 0 : 8  → 8   | lateral: 8×2  = 16
      stage 1 : 8  → 16  | lateral: 8×2  = 16
      stage 2 : 16 → 32  | lateral: 16×2 = 32
      stage 3 : 32 → 32  | lateral: 32×2 = 64

    self.lateral_channels diset saat __init__ dan dibaca oleh SlowPathway.
    """

    def __init__(self, alpha: int = ALPHA):
        super().__init__()
        self.alpha = alpha

        stage_chs = [
            C_SLOW_BASE * 1 // BETA_INV,   # 8
            C_SLOW_BASE * 2 // BETA_INV,   # 16
            C_SLOW_BASE * 4 // BETA_INV,   # 32
            C_SLOW_BASE * 4 // BETA_INV,   # 32
        ]

        # Stem
        self.stem = nn.Sequential(
            Conv3dBnRelu(3, C_FAST_BASE, (5,7,7),
                         stride=(1,2,2), padding=(2,3,3)),
            nn.MaxPool3d((1,3,3), stride=(1,2,2), padding=(0,1,1)),
        )

        # Build stages dan lateral convolutions
        # Lateral: Conv3d temporally-strided (ch → ch*2)
        self.stages    = nn.ModuleList()
        self.lat_convs = nn.ModuleList()

        in_ch = C_FAST_BASE
        self.lateral_channels = []   # disimpan untuk SlowPathway

        for i, out_ch in enumerate(stage_chs):
            lat_out = in_ch * 2
            self.lateral_channels.append(lat_out)

            self.lat_convs.append(nn.Sequential(
                nn.Conv3d(in_ch, lat_out,
                          kernel_size=(alpha, 1, 1),
                          stride=(alpha, 1, 1),
                          bias=False),
                nn.BatchNorm3d(lat_out),
                nn.ReLU(inplace=True),
            ))
            self.stages.append(
                ResBlock3D(in_ch, out_ch,
                           stride_hw=(2 if i < 3 else 1))
            )
            in_ch = out_ch

        self.out_channels = stage_chs[-1]
        self.gap = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        """
        x : (B, 3, T, H, W)
        Returns:
          out      : (B, out_channels)
          laterals : list[4] tensor (B, lat_ch, T//alpha, H_i, W_i)
        """
        x = self.stem(x)
        laterals = []
        for stage, lat_conv in zip(self.stages, self.lat_convs):
            laterals.append(lat_conv(x))   # lateral sebelum downsample
            x = stage(x)
        return self.gap(x).flatten(1), laterals


# ─────────────────────────────────────────────
# Slow Pathway
# ─────────────────────────────────────────────

class SlowPathway(nn.Module):
    """
    Low temporal resolution, high channel count.
    Menerima lateral_channels dari FastPathway untuk channel consistency.

    Channel plan (lateral_channels=[16,16,32,64] dari Fast):
      stem    : 3   → 64
      stage 0 : 64+16=80  → proj → 64  → res → 64
      stage 1 : 64+16=80  → proj → 128 → res → 128
      stage 2 : 128+32=160 → proj → 256 → res → 256
      stage 3 : 256+64=320 → proj → 256 → res → 256
    """

    def __init__(self, lateral_channels: list):
        """
        lateral_channels : list[4] dari FastPathway.lateral_channels
                           Pastikan diambil dari instance Fast yang sama.
        """
        super().__init__()

        stage_chs = [
            C_SLOW_BASE * 1,   # 64
            C_SLOW_BASE * 2,   # 128
            C_SLOW_BASE * 4,   # 256
            C_SLOW_BASE * 4,   # 256
        ]

        # Stem
        self.stem = nn.Sequential(
            Conv3dBnRelu(3, C_SLOW_BASE, (1,7,7),
                         stride=(1,2,2), padding=(0,3,3)),
            nn.MaxPool3d((1,3,3), stride=(1,2,2), padding=(0,1,1)),
        )

        # Build stages — channel dihitung iteratif
        self.lat_fuse = nn.ModuleList()
        self.stages   = nn.ModuleList()

        slow_ch = C_SLOW_BASE
        for i, (out_ch, lat_ch) in enumerate(
                zip(stage_chs, lateral_channels)):
            fuse_in = slow_ch + lat_ch
            # Project concat(slow, lateral) → out_ch
            self.lat_fuse.append(nn.Sequential(
                nn.Conv3d(fuse_in, out_ch, 1, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            ))
            self.stages.append(
                ResBlock3D(out_ch, out_ch,
                           stride_hw=(2 if i < 3 else 1))
            )
            slow_ch = out_ch

        self.out_channels = stage_chs[-1]
        self.gap = nn.AdaptiveAvgPool3d(1)

    def forward(self, x_slow, laterals):
        """
        x_slow   : (B, 3, T_slow, H, W)
        laterals : list[4] dari FastPathway.forward()
        Returns  : (B, out_channels)
        """
        x = self.stem(x_slow)
        for stage, lat_fuse, lat in zip(
                self.stages, self.lat_fuse, laterals):
            # Adaptive pool lateral jika spatial dim berbeda
            if lat.shape[2:] != x.shape[2:]:
                lat = F.adaptive_avg_pool3d(lat, x.shape[2:])
            x_cat = torch.cat([x, lat], dim=1)
            x = stage(lat_fuse(x_cat))
        return self.gap(x).flatten(1)


# ─────────────────────────────────────────────
# B5: SlowFast
# ─────────────────────────────────────────────

class BaselineB5(nn.Module):
    """
    B5 Baseline — SlowFast Network (detection-free).
    Interface identik dengan DASTRPNet dan B1-B4.
    """

    def __init__(
        self,
        base_channels_slow: int   = C_SLOW_BASE,
        alpha:              int   = ALPHA,
        dropout:            float = 0.5,
        T_agg:              int   = None,
        **kwargs,
    ):
        super().__init__()
        self.alpha = alpha

        # Fast → Slow (lateral_channels dari Fast)
        self.fast = FastPathway(alpha=alpha)
        self.slow = SlowPathway(
            lateral_channels=self.fast.lateral_channels
        )

        fusion_dim = self.slow.out_channels + self.fast.out_channels
        self.drop      = nn.Dropout(dropout)
        self.risk_head = MLPRiskHead(in_dim=fusion_dim)

        # Per-frame dummy (replikasi dari clip pred)
        self.frame_risk_proj = nn.Sequential(
            nn.Linear(fusion_dim, 64), nn.ReLU(),
            nn.Linear(64, NUM_RISK),
        )
        self.domain_clf = nn.Linear(fusion_dim, NUM_DOMAINS)

        n_all = sum(p.numel() for p in self.parameters())
        print(f"[BaselineB5] SlowFast (alpha={alpha})")
        print(f"  Fast lateral_channels : {self.fast.lateral_channels}")
        print(f"  Slow out_channels     : {self.slow.out_channels}")
        print(f"  Fast out_channels     : {self.fast.out_channels}")
        print(f"  Fusion dim            : {fusion_dim}")
        print(f"  Total params          : {n_all:,}")

    def reset_tracker(self):
        pass

    def _prepare_inputs(self, clip_images):
        B, T, C, H, W = clip_images.shape
        T_slow = max(1, T // self.alpha)

        x_fast = clip_images.permute(0,2,1,3,4).contiguous()

        idx    = torch.linspace(0, T-1, T_slow).long()
        x_slow = clip_images[:, idx].permute(0,2,1,3,4).contiguous()

        return x_fast, x_slow

    def forward(
        self,
        clip_images:    torch.Tensor,
        domain_ids:     torch.Tensor = None,
        gt_boxes_seq:   list         = None,
        epoch:          int          = 0,
        num_epochs:     int          = 100,
    ) -> dict:
        B, T   = clip_images.shape[:2]
        device = clip_images.device

        x_fast, x_slow = self._prepare_inputs(clip_images)

        fast_out, laterals = self.fast(x_fast)
        slow_out           = self.slow(x_slow, laterals)

        fusion   = self.drop(torch.cat([slow_out, fast_out], dim=1))
        risk_out = self.risk_head(fusion)

        # Per-frame: replikasi prediksi ke T frame
        frame_logit     = self.frame_risk_proj(fusion.detach())
        all_frame_probs = (
            F.softmax(frame_logit, dim=1)
             .unsqueeze(1)
             .expand(-1, T, -1)
             .contiguous()
        )

        domain_logits = self.domain_clf(fusion.detach())

        empty_boxes = [torch.zeros(0,4,device=device) for _ in range(B)]
        empty_mot   = [torch.zeros(0,2,device=device) for _ in range(B)]
        empty_orisk = [torch.zeros(0,3,device=device) for _ in range(B)]
        empty_opred = [torch.zeros(0,device=device,dtype=torch.long)
                       for _ in range(B)]

        return {
            "risk_pred":       risk_out["risk_pred"],
            "risk_prob":       risk_out["risk_prob"],
            "uncertainty":     risk_out["uncertainty"],
            "alpha":           risk_out["alpha"],
            "domain_logits":   domain_logits,
            "all_frame_risks": all_frame_probs,
            "det_boxes":       empty_boxes,
            "motion_logits":   empty_mot,
            "obj_risk_logits": empty_orisk,
            "obj_risk_pred":   empty_opred,
        }
