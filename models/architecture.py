"""
models/architecture.py
======================
DASTRP-Net: Domain-Adaptive Spatio-Temporal Risk Perception Network

Pipeline end-to-end berbasis clip (T frame berurutan):
  Input : (B, T, 3, H, W) clip dari DASTRPClipDataset
  ──────────────────────────────────────────────────────
  Per frame t dalam clip:
    1. YOLO11n pretrained (frozen, 1 kelas "motor")
       → bounding box deteksi + feature map P4
    2. Conditional Batch Normalization per domain
       (UGM=0 / Utara=1 / Ringroad=2)
    3. RoI feature extraction dari P4 per deteksi
    4. VelocityTracker: hitung velocity antar frame berurutan
    5. Spatio-Temporal Graph (GATv2 manual, no torch-geometric)
       → graph embedding per frame
    6. Cross-modal Attention Fusion (frame emb + graph emb)
  ──────────────────────────────────────────────────────
  Setelah semua T frame:
    7. Temporal aggregation (mean pooling T fused embeddings)
    8. Evidential Risk Head → risk_pred + uncertainty
    9. Motion Label Head → gerak/diam per objek (frame terakhir)

  Output:
    risk_pred        : (B,)      0=safe, 1=warning, 2=danger  (frame/clip level)
    risk_prob        : (B, 3)    probabilitas tiap kelas
    uncertainty      : (B,)      0=yakin, 1=tidak yakin
    domain_logits    : (B, 3)    untuk auxiliary domain loss
    det_boxes        : list[B] tensor(N_i, 4)  — frame terakhir
    motion_logits    : list[B] tensor(N_i, 2)  — frame terakhir
    obj_risk_logits  : list[B] tensor(N_i, 3)  — per-objek risk logits
    obj_risk_pred    : list[B] tensor(N_i,)    — per-objek risk prediksi
    all_frame_risks  : (B, T, 3) prediksi risk tiap frame (untuk loss per-frame)

Catatan YOLO:
  - Sudah dilatih dengan 1 kelas "motor" (gerak+diam digabung)
  - Di-freeze sepenuhnya, tidak ikut gradient training DASTRP-Net
  - Saat training: gt_boxes_seq (teacher forcing) dipakai sebagai box
  - Saat inference: YOLO mendeteksi box secara otomatis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


# ─────────────────────────────────────────────
# Konstanta
# ─────────────────────────────────────────────

NUM_DOMAINS  = 3    # UGM=0, Utara=1, Ringroad=2
NUM_RISK     = 3    # safe=0, warning=1, danger=2
NUM_MOTION   = 2    # gerak=0, diam=1

YOLO_FEAT_CH = 256  # channel P4 YOLO11n
FRAME_DIM    = 256  # dimensi frame embedding setelah projection
GRAPH_DIM    = 128  # dimensi graph embedding output GNN
FUSION_DIM   = 256  # dimensi setelah cross-modal fusion


# ─────────────────────────────────────────────
# 1. YOLO Wrapper (frozen)
# ─────────────────────────────────────────────

class YOLODetector(nn.Module):
    """
    Wrapper YOLO11n pretrained (1 kelas "motor") untuk:
      (a) Ekstrak feature map P4 via nn.Sequential backbone (frozen)
      (b) Deteksi bounding box via ultralytics wrapper (lazy-init,
          disimpan di luar module graph agar tidak kena .train() rekursif)

    Kunci desain:
      self.backbone  : nn.Sequential layer backbone+neck yang di-freeze
                       → PyTorch mengelolanya secara normal
      _weights_path  : string biasa, bukan nn.Module
                       → tidak tersentuh .train()/.eval() PyTorch
      _detect_cache  : ultralytics wrapper, disimpan via object.__setattr__
                       → tidak terdaftar sebagai submodule PyTorch
    """

    def __init__(self, yolo_weights: str, conf: float = 0.25, iou: float = 0.45):
        super().__init__()
        from ultralytics import YOLO as _YOLO

        path = Path(yolo_weights)
        if not path.exists():
            raise FileNotFoundError(
                f"YOLO weights tidak ditemukan: {path.resolve()}"
            )
        print(f"[YOLODetector] Loading: {path.resolve()}")

        # Load sementara hanya untuk mengambil layer weights
        _tmp   = _YOLO(str(path))
        _dmodel = _tmp.model   # DetectionModel (nn.Module ultralytics)

        self.conf = conf
        self.iou  = iou

        # ── Probe neck_end ─────────────────────────────────────
        neck_end = self._probe_neck_end(_dmodel)
        print(f"[YOLODetector] neck_end auto-detected: layer {neck_end}")

        # ── Ekstrak layers sebagai nn.Sequential murni ──────────
        # Dengan cara ini TIDAK ada referensi ke DetectionModel/ultralytics
        # di dalam module graph, sehingga model.train() aman
        layers = list(_dmodel.model.children())[:neck_end]
        self.backbone = nn.Sequential(*layers)

        # Freeze semua parameter backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # ── Simpan path sebagai string biasa (bukan nn.Module) ──
        # Ultralytics wrapper dibuat lazy saat detect() pertama dipanggil
        # Disimpan via object.__setattr__ agar tidak masuk module registry
        self._weights_path = str(path)
        object.__setattr__(self, "_detect_cache", None)

        # Hapus referensi sementara
        del _dmodel, _tmp

    @staticmethod
    def _probe_neck_end(det_model) -> int:
        """
        Jalankan forward kecil untuk menemukan indeks layer terakhir
        yang menghasilkan tensor (B, 256, H/16, W/16).
        """
        test_input = torch.zeros(1, 3, 256, 256)
        target_ch  = 256
        target_h   = 256 // 16   # stride 16 → spatial 16×16

        x        = test_input
        best_idx = 12   # fallback untuk YOLO11n berdasarkan hasil probe sebelumnya
        with torch.no_grad():
            for i, layer in enumerate(det_model.model):
                try:
                    x = layer(x)
                except Exception:
                    break
                if (isinstance(x, torch.Tensor)
                        and x.ndim == 4
                        and x.shape[1] == target_ch
                        and x.shape[2] == target_h):
                    best_idx = i + 1   # +1: loop backbone pakai [:neck_end]
        return best_idx

    def _get_detect_wrapper(self):
        """
        Lazy-init ultralytics YOLO wrapper untuk detect().
        Dikembalikan dari cache jika sudah ada.
        """
        cache = object.__getattribute__(self, "_detect_cache")
        if cache is None:
            from ultralytics import YOLO as _YOLO
            cache = _YOLO(self._weights_path)
            object.__setattr__(self, "_detect_cache", cache)
        return cache

    # ── Feature extraction ──────────────────────────────────────

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor):
        """
        Forward pass backbone untuk mendapat P4 feature map.

        Args:
            images : (B, 3, H, W) normalized ImageNet

        Returns:
            frame_feat : (B, 256, H/16, W/16)
            frame_emb  : (B, 256)  global average pooled
        """
        # Paksa fp32 meskipun dipanggil dalam autocast context
        # backbone frozen → lebih stabil di fp32
        with torch.amp.autocast("cuda", enabled=False):
            x = self.backbone(images.float())
        frame_feat = x
        frame_emb  = x.mean(dim=[2, 3])
        return frame_feat, frame_emb

    # ── Detection ───────────────────────────────────────────────

    @torch.no_grad()
    def detect(self, images: torch.Tensor) -> list:
        """
        Deteksi bounding box motor via ultralytics wrapper.
        Dipanggil hanya saat inference (gt_boxes_seq=None).

        Returns:
            list of (N_i, 4) tensor [cx, cy, w, h] normalized
        """
        mean    = torch.tensor([0.485, 0.456, 0.406],
                               device=images.device).view(1, 3, 1, 1)
        std     = torch.tensor([0.229, 0.224, 0.225],
                               device=images.device).view(1, 3, 1, 1)
        imgs_u8 = ((images * std + mean) * 255).clamp(0, 255).byte()

        wrapper = self._get_detect_wrapper()
        results = []
        for i in range(images.size(0)):
            img_np = imgs_u8[i].permute(1, 2, 0).cpu().numpy()
            res    = wrapper.predict(
                img_np,
                conf=self.conf,
                iou=self.iou,
                verbose=False,
            )[0]
            results.append(res.boxes.xywhn.to(images.device))
        return results


# ─────────────────────────────────────────────
# 2. Conditional Batch Normalization
# ─────────────────────────────────────────────

class ConditionalBN(nn.Module):
    """
    Satu pasang (γ, β) per domain — domain-specific normalization.

    Menangani distribusi statistik berbeda antara:
      UGM (kampus terstruktur) / Utara (jalan tanpa pemisah) /
      Ringroad (jalan lebar, occluded, blur).
    """

    def __init__(self, num_features: int, num_domains: int = NUM_DOMAINS):
        super().__init__()
        self.bn    = nn.BatchNorm1d(num_features, affine=False)
        self.gamma = nn.Embedding(num_domains, num_features)
        self.beta  = nn.Embedding(num_domains, num_features)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x: torch.Tensor, domain_ids: torch.Tensor) -> torch.Tensor:
        """
        x          : (B, C)
        domain_ids : (B,) long  —  0 / 1 / 2
        """
        # BatchNorm1d butuh B > 1 saat training
        if x.size(0) == 1 and self.training:
            out = x   # skip BN untuk single sample
        else:
            out = self.bn(x)
        return self.gamma(domain_ids) * out + self.beta(domain_ids)


class SceneClassifier(nn.Module):
    """
    Prediksi domain dari frame embedding.
    Dipakai saat inference (domain_ids tidak tersedia dari luar).
    """

    def __init__(self, in_dim: int = FRAME_DIM, num_domains: int = NUM_DOMAINS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),    nn.ReLU(),
            nn.Linear(64, num_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, num_domains) logits


# ─────────────────────────────────────────────
# 3. RoI Feature Extraction
# ─────────────────────────────────────────────

class RoIExtractor(nn.Module):
    """
    Crop P4 feature map per bounding box, pool ke (4,4), project ke vektor.
    Menghasilkan per-object feature untuk GNN node dan motion head.
    """

    def __init__(self, in_channels: int = YOLO_FEAT_CH, out_dim: int = FRAME_DIM):
        super().__init__()
        self._out_dim = out_dim
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 4 * 4, out_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        feat_map: torch.Tensor,   # (1, C, H', W') — single frame
        boxes:    torch.Tensor,   # (N, 4) [cx, cy, w, h] normalized
    ) -> torch.Tensor:            # (N, out_dim)
        N = boxes.size(0)
        if N == 0:
            return torch.zeros(0, self._out_dim, device=feat_map.device)

        _, C, fH, fW = feat_map.shape
        feats = []
        for box in boxes:
            cx, cy, w, h = box.tolist()
            x1 = max(0,  int((cx - w / 2) * fW))
            y1 = max(0,  int((cy - h / 2) * fH))
            x2 = min(fW, int((cx + w / 2) * fW))
            y2 = min(fH, int((cy + h / 2) * fH))
            if x2 <= x1 or y2 <= y1:
                x1, y1, x2, y2 = 0, 0, 1, 1
            crop   = feat_map[:, :, y1:y2, x1:x2]
            pooled = self.pool(crop)
            feats.append(self.proj(pooled))
        return torch.cat(feats, dim=0)


# ─────────────────────────────────────────────
# 4. Velocity Tracker (stateful)
# ─────────────────────────────────────────────

class VelocityTracker:
    """
    Estimasi velocity per objek via nearest-center matching antar frame.
    Stateful: menyimpan boxes frame sebelumnya.
    Di-reset setiap awal clip baru.
    """

    def __init__(self):
        self.prev_boxes = None

    def reset(self):
        self.prev_boxes = None

    @torch.no_grad()
    def compute(self, boxes: torch.Tensor) -> torch.Tensor:
        N       = boxes.size(0)
        boxes_f = boxes.float()
        vels    = torch.zeros(N, 2, device=boxes.device, dtype=torch.float32)

        if (self.prev_boxes is not None
                and N > 0
                and self.prev_boxes.size(0) > 0):
            curr_c = boxes_f[:, :2]
            prev_c = self.prev_boxes.float()[:, :2]
            dists  = torch.cdist(curr_c, prev_c)
            match  = dists.argmin(dim=1)
            vels   = curr_c - prev_c[match]

        self.prev_boxes = boxes.detach().clone()
        return vels


# ─────────────────────────────────────────────
# 5. Motion Graph Builder + GATv2 GNN
# ─────────────────────────────────────────────

class MotionGraphBuilder:

    @staticmethod
    def build(
        boxes:       torch.Tensor,
        vels:        torch.Tensor,
        roi_feats:   torch.Tensor,
        dist_thresh: float = 0.3,
    ) -> tuple:
        N  = boxes.size(0)
        cx = boxes[:, 0:1]; cy = boxes[:, 1:2]
        w  = boxes[:, 2:3]; h  = boxes[:, 3:4]
        node_x = torch.cat([roi_feats, cx, cy, w, h, vels], dim=1)

        if N <= 1:
            ei = torch.zeros(2, 0, dtype=torch.long, device=boxes.device)
            ea = torch.zeros(0, 2, device=boxes.device)
            return node_x, ei, ea

        centers = boxes[:, :2]
        dists   = torch.cdist(centers.float(), centers.float())
        mask    = (dists < dist_thresh) & ~torch.eye(
            N, dtype=torch.bool, device=boxes.device)
        src, dst = mask.nonzero(as_tuple=True)

        if src.size(0) == 0:
            ei = torch.zeros(2, 0, dtype=torch.long, device=boxes.device)
            ea = torch.zeros(0, 2, device=boxes.device)
            return node_x, ei, ea

        edge_index = torch.stack([src, dst], dim=0)
        dist_e     = dists[src, dst].unsqueeze(1)
        vel_diff   = (vels[src] - vels[dst]).norm(dim=1, keepdim=True)
        edge_attr  = torch.cat([dist_e, vel_diff], dim=1)
        return node_x, edge_index, edge_attr


class SpatioTemporalGNN(nn.Module):

    def __init__(
        self,
        node_dim: int,
        hidden:   int = 128,
        out_dim:  int = GRAPH_DIM,
        heads:    int = 4,
    ):
        super().__init__()
        self.heads   = heads
        self.hidden  = hidden
        self.out_dim = out_dim
        Hd           = hidden * heads

        self.Wq1 = nn.Linear(node_dim, Hd, bias=False)
        self.Wk1 = nn.Linear(node_dim, Hd, bias=False)
        self.Wv1 = nn.Linear(node_dim, Hd, bias=False)
        self.We1 = nn.Linear(2, Hd, bias=False)
        self.a1  = nn.Parameter(torch.empty(heads, hidden))
        self.n1  = nn.LayerNorm(Hd)
        nn.init.xavier_uniform_(self.a1.unsqueeze(0))

        self.Wq2 = nn.Linear(Hd, Hd, bias=False)
        self.Wk2 = nn.Linear(Hd, Hd, bias=False)
        self.Wv2 = nn.Linear(Hd, Hd, bias=False)
        self.We2 = nn.Linear(2, Hd, bias=False)
        self.a2  = nn.Parameter(torch.empty(heads, hidden))
        self.n2  = nn.LayerNorm(Hd)
        nn.init.xavier_uniform_(self.a2.unsqueeze(0))

        self.out_proj = nn.Sequential(nn.Linear(Hd, out_dim), nn.ReLU())

    def _gat(self, x, ei, ea, Wq, Wk, Wv, We, a, norm):
        N = x.size(0)
        H = self.heads
        d = self.hidden
        Q = Wq(x); K = Wk(x); V = Wv(x)

        if ei.size(1) == 0:
            return F.elu(norm(Q))

        src, dst = ei
        Ef    = We(ea).view(-1, H, d)
        Qs    = Q[src].view(-1, H, d)
        Kd    = K[dst].view(-1, H, d)
        score = (F.leaky_relu(Qs + Kd + Ef, 0.2) * a).sum(-1)

        s_exp = score.exp()
        denom = torch.zeros(N, H, device=x.device)
        denom.scatter_add_(0, dst.unsqueeze(1).expand_as(s_exp), s_exp)
        alpha_e = s_exp / denom[dst].clamp(min=1e-8)

        Vs  = V[src].view(-1, H, d)
        agg = torch.zeros(N, H, d, device=x.device)
        agg.scatter_add_(
            0,
            dst.view(-1, 1, 1).expand_as(Vs),
            alpha_e.unsqueeze(-1) * Vs,
        )
        return F.elu(norm(agg.view(N, H * d)))

    def forward(self, node_x, edge_index, edge_attr):
        if node_x.size(0) == 0:
            return torch.zeros(self.out_dim, device=node_x.device)
        x = self._gat(node_x, edge_index, edge_attr,
                      self.Wq1, self.Wk1, self.Wv1, self.We1, self.a1, self.n1)
        x = self._gat(x, edge_index, edge_attr,
                      self.Wq2, self.Wk2, self.Wv2, self.We2, self.a2, self.n2)
        return self.out_proj(x.mean(dim=0))


# ─────────────────────────────────────────────
# 6. Cross-Modal Attention Fusion
# ─────────────────────────────────────────────

class CrossModalFusion(nn.Module):

    def __init__(
        self,
        frame_dim: int = FRAME_DIM,
        graph_dim: int = GRAPH_DIM,
        out_dim:   int = FUSION_DIM,
        heads:     int = 4,
    ):
        super().__init__()
        self.q_proj = nn.Linear(frame_dim, out_dim)
        self.k_proj = nn.Linear(graph_dim, out_dim)
        self.v_proj = nn.Linear(graph_dim, out_dim)
        self.attn   = nn.MultiheadAttention(out_dim, heads, batch_first=True)
        self.norm   = nn.LayerNorm(out_dim)
        self.ff     = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2), nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, frame_emb, graph_emb):
        Q = self.q_proj(frame_emb).unsqueeze(1)
        K = self.k_proj(graph_emb).unsqueeze(1)
        V = self.v_proj(graph_emb).unsqueeze(1)
        z, _ = self.attn(Q, K, V)
        z    = z.squeeze(1)
        res  = self.norm(z + Q.squeeze(1))
        return res + self.ff(res)


# ─────────────────────────────────────────────
# 7. Evidential Risk Head
# ─────────────────────────────────────────────

class EvidentialRiskHead(nn.Module):

    def __init__(self, in_dim: int = FUSION_DIM, num_classes: int = NUM_RISK):
        super().__init__()
        self.K   = num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),    nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        evidence  = F.softplus(self.net(x))
        alpha     = evidence + 1
        S         = alpha.sum(1, keepdim=True)
        risk_prob = alpha / S
        return {
            "alpha":       alpha,
            "risk_prob":   risk_prob,
            "risk_pred":   risk_prob.argmax(1),
            "uncertainty": self.K / S.squeeze(1),
        }

    @staticmethod
    def edl_loss(alpha, labels, epoch, num_epochs, lam=1.0):
        K  = alpha.size(1)
        S  = alpha.sum(1, keepdim=True)
        y  = F.one_hot(labels, K).float()
        l_nll   = (y * (S.log() - alpha.log())).sum(1)
        w       = min(1.0, epoch / max(1, int(num_epochs * 0.3)))
        a_tilde = y + (1 - y) * alpha
        S_t     = a_tilde.sum(1, keepdim=True)
        lgG_K   = torch.lgamma(torch.tensor(float(K), device=alpha.device))
        kl = (torch.lgamma(S_t) - lgG_K
              - torch.lgamma(a_tilde).sum(1, keepdim=True)
              + ((a_tilde - 1) * (a_tilde.digamma() - S_t.digamma()))
                .sum(1, keepdim=True)).squeeze(1)
        return (l_nll + w * lam * kl).mean()


# ─────────────────────────────────────────────
# 8. Motion Label Head
# ─────────────────────────────────────────────

class MotionLabelHead(nn.Module):

    def __init__(self, in_dim: int = FRAME_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, NUM_MOTION),
        )

    def forward(self, roi_feats):
        if roi_feats.size(0) == 0:
            return torch.zeros(0, NUM_MOTION, device=roi_feats.device)
        return self.net(roi_feats)


# ─────────────────────────────────────────────
# 9. Object Risk Head (per-objek)
# ─────────────────────────────────────────────

class ObjectRiskHead(nn.Module):
    """
    Klasifikasi risk level per objek dari RoI feature.

    Berbeda dari EvidentialRiskHead (yang beroperasi di level clip/frame),
    head ini menghasilkan prediksi risk untuk setiap motor yang terdeteksi
    secara individual, menggunakan RoI feature dari P4 backbone.

    Input  : (N, FRAME_DIM) — RoI features per objek
    Output : (N, 3) logits  — 0=safe, 1=warning, 2=danger

    Desain:
      - Lebih dalam dari MotionLabelHead karena risk lebih kompleks dari
        gerak/diam: perlu memahami konteks visual objek (posisi, ukuran,
        kecepatan implisit dari RoI)
      - Menggunakan ReLU + Dropout untuk regularisasi mengingat label
        per-objek jauh lebih imbalanced dari label frame
    """

    def __init__(self, in_dim: int = FRAME_DIM, num_classes: int = NUM_RISK):
        super().__init__()
        self.K   = num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128,  64),   nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, roi_feats: torch.Tensor) -> torch.Tensor:
        """
        roi_feats : (N, FRAME_DIM)
        Returns   : (N, 3) logits — belum di-softmax
        """
        if roi_feats.size(0) == 0:
            return torch.zeros(0, self.K, device=roi_feats.device)
        return self.net(roi_feats)


# ─────────────────────────────────────────────
# 9b. Softmax Risk Head (ablation A4: no_edl)
# ─────────────────────────────────────────────

class _SoftmaxRiskHead(nn.Module):
    """
    Ablation A4: replaces EvidentialRiskHead with a standard softmax
    classifier trained with cross-entropy (see compute_loss in
    scripts/train.py). Output keys match EvidentialRiskHead so the rest
    of the pipeline is unchanged; `uncertainty` is a constant 1/K
    (non-informative) and `alpha` is a dummy (prob + 1).
    """

    def __init__(self, in_dim: int = FUSION_DIM, num_classes: int = NUM_RISK):
        super().__init__()
        self.K   = num_classes
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),    nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> dict:
        logits    = self.net(x)
        risk_prob = F.softmax(logits, dim=1)
        return {
            "logits":      logits,
            "alpha":       risk_prob + 1,
            "risk_prob":   risk_prob,
            "risk_pred":   risk_prob.argmax(1),
            "uncertainty": torch.full((x.size(0),), 1.0 / self.K,
                                      device=x.device),
        }


# ─────────────────────────────────────────────
# 10. DASTRP-Net (model utama)
# ─────────────────────────────────────────────

class DASTRPNet(nn.Module):

    def __init__(
        self,
        yolo_weights:  str,
        yolo_conf:     float = 0.25,
        yolo_iou:      float = 0.45,
        dist_thresh:   float = 0.3,
        T_agg:         int   = None,
        ablation:      str   = None,
    ):
        """
        ablation : None (full model) or one of
                   "no_gnn"      (A1) graph embedding replaced by zeros
                   "no_cbn"      (A2) ConditionalBN -> standard BatchNorm1d
                   "no_fusion"   (A3) CrossModalFusion -> concat + linear
                   "no_edl"      (A4) EvidentialRiskHead -> softmax + CE
                   "no_velocity" (A5) velocity vectors set to zero
                   "no_obj_risk" (A6) ObjectRiskHead and its loss removed
        """
        super().__init__()
        valid = {None, "no_gnn", "no_cbn", "no_fusion",
                 "no_edl", "no_velocity", "no_obj_risk"}
        if ablation not in valid:
            raise ValueError(f"Unknown ablation '{ablation}'")
        self.dist_thresh = dist_thresh
        self.T_agg       = T_agg
        self.ablation    = ablation

        self.yolo          = YOLODetector(yolo_weights, yolo_conf, yolo_iou)
        self.frame_proj    = nn.Sequential(
            nn.Linear(YOLO_FEAT_CH, FRAME_DIM), nn.ReLU()
        )
        self.scene_clf     = SceneClassifier(in_dim=FRAME_DIM)
        # A2: no_cbn -> standard BN without domain conditioning
        if ablation == "no_cbn":
            self.domain_norm = nn.BatchNorm1d(FRAME_DIM)
        else:
            self.domain_norm = ConditionalBN(FRAME_DIM)
        self.roi_extractor  = RoIExtractor(in_channels=YOLO_FEAT_CH, out_dim=FRAME_DIM)
        self.motion_head    = MotionLabelHead(in_dim=FRAME_DIM)
        # A6: no_obj_risk -> no ObjectRiskHead
        if ablation != "no_obj_risk":
            self.obj_risk_head = ObjectRiskHead(in_dim=FRAME_DIM)

        # A1: no_gnn -> no graph encoder
        node_dim = FRAME_DIM + 6
        if ablation != "no_gnn":
            self.gnn = SpatioTemporalGNN(node_dim=node_dim, hidden=128,
                                         out_dim=GRAPH_DIM, heads=4)
        # A3: no_fusion -> concat + linear projection
        if ablation == "no_fusion":
            self.fusion = nn.Sequential(
                nn.Linear(FRAME_DIM + GRAPH_DIM, FUSION_DIM), nn.ReLU()
            )
        else:
            self.fusion = CrossModalFusion(frame_dim=FRAME_DIM, graph_dim=GRAPH_DIM,
                                           out_dim=FUSION_DIM)
        # A4: no_edl -> softmax classifier
        if ablation == "no_edl":
            self.risk_head = _SoftmaxRiskHead(in_dim=FUSION_DIM)
        else:
            self.risk_head = EvidentialRiskHead(in_dim=FUSION_DIM)
        self._vel_tracker = VelocityTracker()

    def reset_tracker(self):
        self._vel_tracker.reset()

    def _process_frame(self, images_t, domain_ids, gt_boxes_t, device):
        B = images_t.size(0)

        # Step 1 — backbone (fp32, no_grad)
        frame_feat, frame_emb_raw = self.yolo.extract_features(images_t)

        # Step 2 — project ke FRAME_DIM (trainable, cast ke dtype images)
        frame_emb = self.frame_proj(frame_emb_raw.to(images_t.dtype))

        # Step 3 — bounding boxes
        if gt_boxes_t is not None:
            boxes_list = [b.to(device) for b in gt_boxes_t]
        else:
            boxes_list = self.yolo.detect(images_t)

        # Step 4 — domain normalization
        domain_logits = self.scene_clf(frame_emb)
        eff_domain    = (domain_logits.argmax(1)
                         if domain_ids is None else domain_ids)
        if self.ablation == "no_cbn":
            frame_emb = self.domain_norm(frame_emb)
        else:
            frame_emb = self.domain_norm(frame_emb, eff_domain)

        # Step 5 — RoI + velocity + graph per image dalam batch
        graph_embs    = []
        motion_logs   = []
        obj_risk_logs = []   # per-objek risk logits
        roi_list      = []

        for i in range(B):
            boxes     = boxes_list[i]
            roi_feats = self.roi_extractor(frame_feat[i:i+1], boxes)
            roi_list.append(roi_feats)

            # Motion label per objek (gerak/diam)
            motion_logs.append(self.motion_head(roi_feats))

            # Per-objek risk level (safe/warning/danger)
            # Menggunakan RoI feature yang sama — tidak ada overhead backbone
            if self.ablation != "no_obj_risk":
                obj_risk_logs.append(self.obj_risk_head(roi_feats))
            else:
                obj_risk_logs.append(
                    torch.zeros(roi_feats.size(0), NUM_RISK, device=device))

            # A5: no_velocity -> zero velocity vectors
            #     (tracker still updated so state handling is identical)
            vels = self._vel_tracker.compute(boxes)
            if self.ablation == "no_velocity":
                vels = torch.zeros_like(vels)

            # A1: no_gnn -> zero graph embedding
            if self.ablation == "no_gnn":
                graph_embs.append(torch.zeros(GRAPH_DIM, device=device,
                                              dtype=frame_emb.dtype))
            else:
                node_x, ei, ea = MotionGraphBuilder.build(
                    boxes, vels, roi_feats, self.dist_thresh)
                graph_embs.append(self.gnn(node_x, ei, ea))

        graph_emb = torch.stack(graph_embs, dim=0)

        # Step 6 — fusion (A3: concat + linear)
        if self.ablation == "no_fusion":
            fused = self.fusion(torch.cat([frame_emb, graph_emb.to(frame_emb.dtype)], dim=1))
        else:
            fused = self.fusion(frame_emb, graph_emb)
        return fused, domain_logits, boxes_list, motion_logs, obj_risk_logs, roi_list

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

        self._vel_tracker.reset()

        all_fused        = []
        all_frame_probs  = []
        domain_logits_t0 = None
        last_boxes       = None
        last_motion      = None
        last_obj_risk    = None   # per-objek risk — frame terakhir

        for t in range(T):
            images_t   = clip_images[:, t]
            gt_boxes_t = ([gt_boxes_seq[b][t] for b in range(B)]
                          if gt_boxes_seq is not None else None)

            fused_t, dom_logits_t, boxes_t, motion_t, obj_risk_t, _ = \
                self._process_frame(images_t, domain_ids, gt_boxes_t, device)

            all_fused.append(fused_t)

            if t < T - 1:
                with torch.no_grad():
                    risk_t = self.risk_head(fused_t)
            else:
                risk_t = self.risk_head(fused_t)
            all_frame_probs.append(risk_t["risk_prob"].detach())

            if t == 0:
                domain_logits_t0 = dom_logits_t
            last_boxes    = boxes_t
            last_motion   = motion_t
            last_obj_risk = obj_risk_t   # timpa tiap frame, simpan terakhir

        fused_seq = torch.stack(all_fused, dim=1)
        agg = (fused_seq[:, -self.T_agg:].mean(dim=1)
               if self.T_agg is not None
               else fused_seq.mean(dim=1))

        risk_out        = self.risk_head(agg)
        all_frame_risks = torch.stack(all_frame_probs, dim=1)

        # Per-objek risk prediksi (argmax dari logits)
        obj_risk_pred = [
            logits.argmax(1) if logits.size(0) > 0
            else torch.zeros(0, dtype=torch.long, device=agg.device)
            for logits in last_obj_risk
        ]

        return {
            # Frame/clip level
            "risk_pred":       risk_out["risk_pred"],
            "risk_prob":       risk_out["risk_prob"],
            "uncertainty":     risk_out["uncertainty"],
            "alpha":           risk_out["alpha"],
            "risk_logits":     risk_out.get("logits"),   # only for no_edl
            "domain_logits":   domain_logits_t0,
            "all_frame_risks": all_frame_risks,
            # Per-frame terakhir (detection + per-objek label)
            "det_boxes":       last_boxes,
            "motion_logits":   last_motion,
            "obj_risk_logits": last_obj_risk,   # list[B] (N_i, 3)
            "obj_risk_pred":   obj_risk_pred,   # list[B] (N_i,)
        }
