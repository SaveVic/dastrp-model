"""
models/baseline_b4.py
=====================
B4 Baseline: Scene-Graph GCN + LSTM (Yu et al., IEEE T-ITS 2021)

Re-implementasi yang diadaptasi dari:
  Yu, S.-Y. et al., "Scene-Graph Augmented Data-Driven Risk
  Assessment of Autonomous Vehicle Decisions," IEEE Trans.
  Intell. Transp. Syst., vol. 23, no. 7, pp. 6567-6580, 2021.
  DOI: 10.1109/TITS.2021.3072583

Adaptasi untuk dataset ini:
  - Backbone: YOLO11n (frozen) menggantikan Faster R-CNN
  - Task: three-class risk (Safe/Warning/Danger) per clip
    menggantikan binary collision risk per maneuver
  - Scene graph: motor-motor relation graph (distance, velocity)
    menggantikan vehicle-vehicle interaction graph
  - Output: kompatibel dengan train.py dan evaluate.py

Pipeline:
  Input : (B, T, 3, H, W) clip
  ──────────────────────────────────────────────────────
  Per frame t:
    1. YOLO11n backbone (frozen) → P4 feature map
    2. RoI extraction per deteksi → object features (N_t, D_obj)
    3. Scene graph construction:
         Node: object feature + bbox geometry + motion label
         Edge: relation antara semua pasangan (fully connected)
               dengan edge weight berdasarkan spatial distance
    4. GCN (Graph Convolutional Network) → refined node features
    5. Graph-level pooling → scene embedding (B, D_scene)
  ──────────────────────────────────────────────────────
  Across T frames:
    6. LSTM → temporal aggregation (B, D_scene) × T → (B, D_lstm)
    7. MLP Risk Head → risk_pred (softmax, bukan EDL)

Perbedaan dari DASTRPNet:
  ✓ Ada scene graph (GCN) → ada interaksi antar motor
  ✓ Ada temporal modeling (LSTM)
  ✗ GCN symmetric (bukan GATv2 attention-based)
  ✗ Tidak ada domain adaptation (tidak ada CBN)
  ✗ Tidak ada cross-modal attention fusion
  ✗ Tidak ada EDL uncertainty
  ✗ Tidak ada velocity-augmented node features
  ✗ Tidak ada per-objek risk head

Perbedaan dari DASTRPNet (GATv2):
  GCN  → symmetric aggregation tanpa attention weight per edge
  GATv2→ anisotropic attention, edge-conditioned attention score
         berbeda per pasangan (i,j)

Output dict: kompatibel penuh dengan train.py dan evaluate.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from architecture import (
    YOLODetector, RoIExtractor, VelocityTracker
)
from baseline_b1 import MLPRiskHead

NUM_RISK    = 3
NUM_DOMAINS = 3
YOLO_CH     = 256   # channel P4 YOLO11n
OBJ_DIM     = 256   # dimensi object feature (dari RoI)
RELATION_DIM = 64   # dimensi relation feature antar pasangan objek
GCN_DIM     = 256   # dimensi output GCN per node
SCENE_DIM   = 256   # dimensi scene graph embedding (setelah pooling)
LSTM_DIM    = 256   # dimensi LSTM hidden state
FUSION_DIM  = 256   # dimensi output sebelum risk head


# ─────────────────────────────────────────────
# Relation Feature Encoder
# ─────────────────────────────────────────────

class RelationEncoder(nn.Module):
    """
    Encode relasi antar dua objek dari fitur geometri mereka.

    Input untuk pasangan (i, j):
      [cx_i, cy_i, w_i, h_i, cx_j, cy_j, w_j, h_j,
       d_ij (euclidean), Δvx, Δvy]
    → relation feature (RELATION_DIM,)

    Berbeda dari GATv2 yang menggunakan edge_attr 2-dim sederhana,
    RelationEncoder mempelajari relasi yang lebih kaya termasuk
    geometri relatif kedua objek.
    """

    def __init__(self, in_dim: int = 11, out_dim: int = RELATION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, out_dim), nn.ReLU(),
        )

    def forward(
        self,
        boxes_i: torch.Tensor,   # (E, 4) [cx, cy, w, h] untuk source
        boxes_j: torch.Tensor,   # (E, 4) [cx, cy, w, h] untuk target
        vels_i:  torch.Tensor,   # (E, 2) velocity source
        vels_j:  torch.Tensor,   # (E, 2) velocity target
    ) -> torch.Tensor:           # (E, out_dim)
        """
        Encode semua pasangan edge sekaligus.
        E = jumlah edge dalam scene graph.
        """
        # Euclidean distance antar center
        d_ij = (boxes_i[:, :2] - boxes_j[:, :2]).norm(dim=1, keepdim=True)
        # Relative velocity
        dv   = vels_i - vels_j   # (E, 2)
        # Concatenate semua fitur relasi
        x = torch.cat([boxes_i, boxes_j, d_ij, dv], dim=1)  # (E, 11)
        return self.net(x)


# ─────────────────────────────────────────────
# Graph Convolutional Network (GCN)
# ─────────────────────────────────────────────

class SceneGCN(nn.Module):
    """
    2-layer Graph Convolutional Network (Kipf & Welling 2017)
    dengan relation-aware aggregation.

    Berbeda dari GATv2 yang menggunakan learnable attention weights,
    GCN menggunakan aggregasi symmetric dengan normalisasi degree:

      H^(l+1) = σ( D̃^{-1/2} Ã D̃^{-1/2} H^(l) W^(l) + R^(l) )

    Di mana:
      Ã = A + I (adjacency + self-loop)
      D̃ = degree matrix dari Ã
      R^(l) = relasi yang diproyeksikan ke ruang node (relation injection)

    Relation injection memungkinkan GCN memanfaatkan fitur edge
    tanpa attention mechanism, mendekati arsitektur Yu et al. 2021.

    Args:
        node_dim    : dimensi input node feature
        relation_dim: dimensi relation feature dari RelationEncoder
        hidden_dim  : dimensi hidden layer GCN
        out_dim     : dimensi output per node
        n_layers    : jumlah GCN layers (default: 2)
        dropout     : dropout antar layer
    """

    def __init__(
        self,
        node_dim:     int   = OBJ_DIM,
        relation_dim: int   = RELATION_DIM,
        hidden_dim:   int   = GCN_DIM,
        out_dim:      int   = GCN_DIM,
        n_layers:     int   = 2,
        dropout:      float = 0.3,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.dropout  = dropout

        # GCN weight matrices W per layer
        dims = [node_dim] + [hidden_dim] * (n_layers - 1) + [out_dim]
        self.W = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1], bias=False)
            for i in range(n_layers)
        ])

        # Relation injection: project relation feat ke node space per layer
        self.R = nn.ModuleList([
            nn.Linear(relation_dim, dims[i+1], bias=False)
            for i in range(n_layers)
        ])

        # Layer normalization per layer
        self.norms = nn.ModuleList([
            nn.LayerNorm(dims[i+1]) for i in range(n_layers)
        ])

    def forward(
        self,
        node_x:      torch.Tensor,   # (N, node_dim)
        adj:         torch.Tensor,   # (N, N) normalized adjacency
        edge_index:  torch.Tensor,   # (2, E) untuk relation injection
        edge_feat:   torch.Tensor,   # (E, relation_dim)
    ) -> torch.Tensor:               # (N, out_dim)
        """
        Forward pass GCN dengan relation injection.

        adj    : D̃^{-1/2} Ã D̃^{-1/2} — symmetric normalized adjacency
        edge_index: (src, dst) pairs untuk mengakumulasi relation feat
        edge_feat : (E, relation_dim) dari RelationEncoder
        """
        N = node_x.size(0)
        h = node_x

        for l in range(self.n_layers):
            # Pastikan adj dan h dtype sama sebelum matmul
            h_agg = adj.to(h.dtype) @ self.W[l](h)   # (N, hidden_dim)

            # Relation injection: akumulasi relation feat ke node dst
            if edge_feat.size(0) > 0:
                src, dst = edge_index
                r_proj   = self.R[l](edge_feat).to(h.dtype)  # cast ke dtype h
                r_agg    = torch.zeros(N, r_proj.size(1),
                                       device=h.device, dtype=h.dtype)
                r_agg.scatter_add_(
                    0,
                    dst.unsqueeze(1).expand_as(r_proj),
                    r_proj
                )
                deg = torch.zeros(N, device=h.device, dtype=h.dtype)
                deg.scatter_add_(0, dst,
                                 torch.ones(dst.size(0),
                                            device=h.device, dtype=h.dtype))
                r_agg = r_agg / (deg.unsqueeze(1).clamp(min=1))
                h_agg = h_agg + r_agg

            h = self.norms[l](F.relu(h_agg))
            if l < self.n_layers - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)

        return h   # (N, out_dim)


# ─────────────────────────────────────────────
# Scene Graph Builder
# ─────────────────────────────────────────────

def build_scene_graph(
    boxes:        torch.Tensor,   # (N, 4) [cx, cy, w, h]
    vels:         torch.Tensor,   # (N, 2)
    motion_label: torch.Tensor,   # (N,) 0=gerak, 1=diam
    roi_feats:    torch.Tensor,   # (N, OBJ_DIM)
    device:       torch.device,
) -> tuple:
    """
    Bangun scene graph dari deteksi satu frame.

    Node feature: [roi_feat, cx, cy, w, h, vx, vy, motion_label]
                → (N, OBJ_DIM + 7)

    Edge: fully connected (semua pasangan i≠j) berbeda dari DASTRPNet
    yang menggunakan proximity threshold. Fully connected mengikuti
    Yu et al. 2021 yang menggunakan scene graph lengkap.

    Returns:
        node_x       : (N, OBJ_DIM+7)
        adj_norm     : (N, N) symmetric normalized adjacency
        edge_index   : (2, E) — E = N*(N-1)
        edge_raw     : (E, 4+4+2) input untuk RelationEncoder
                       [boxes_i, boxes_j, vels_i, vels_j] sudah split
        boxes_i/j    : (E, 4) untuk RelationEncoder
        vels_i/j     : (E, 2) untuk RelationEncoder
    """
    N = boxes.size(0)

    # ── Node features ─────────────────────────────────────────
    mot = motion_label.float().unsqueeze(1)   # (N, 1)
    node_x = torch.cat([roi_feats,
                         boxes,        # cx, cy, w, h
                         vels,         # vx, vy
                         mot],         # motion label
                        dim=1)         # (N, OBJ_DIM+7)

    if N <= 1:
        # Single node — tidak ada edge
        adj_norm   = torch.eye(N, device=device)
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        empty      = torch.zeros(0, 4, device=device)
        return node_x, adj_norm, edge_index, empty, empty, empty, empty

    # ── Fully connected edge index ────────────────────────────
    rows, cols = [], []
    for i in range(N):
        for j in range(N):
            if i != j:
                rows.append(i)
                cols.append(j)
    src = torch.tensor(rows, dtype=torch.long, device=device)
    dst = torch.tensor(cols, dtype=torch.long, device=device)
    edge_index = torch.stack([src, dst], dim=0)   # (2, E)

    # ── Normalized adjacency ──────────────────────────────────
    # Selalu float32 — GCN akan cast ke dtype h saat forward
    A = torch.ones(N, N, device=device, dtype=torch.float32) \
        - torch.eye(N, device=device, dtype=torch.float32)
    A = A + torch.eye(N, device=device, dtype=torch.float32)
    deg      = A.sum(1, keepdim=True)
    deg_inv  = deg.pow(-0.5)
    adj_norm = deg_inv * A * deg_inv.T

    # ── Edge raw features untuk RelationEncoder ───────────────
    boxes_i = boxes[src]   # (E, 4)
    boxes_j = boxes[dst]
    vels_i  = vels[src]    # (E, 2)
    vels_j  = vels[dst]

    return node_x, adj_norm, edge_index, boxes_i, boxes_j, vels_i, vels_j


# ─────────────────────────────────────────────
# B4 Model: Scene-Graph GCN + LSTM
# ─────────────────────────────────────────────

class BaselineB4(nn.Module):
    """
    B4 Baseline — Scene-Graph GCN + LSTM.

    Adaptasi dari Yu et al. (IEEE T-ITS 2021) untuk dataset ini:
    - Backbone: YOLO11n frozen (menggantikan Faster R-CNN)
    - Scene graph: motor-motor interaction (fully connected)
    - GCN 2-layer dengan relation injection
    - LSTM untuk temporal aggregation T frame
    - Softmax risk head (3 kelas: Safe/Warning/Danger)

    Perbandingan dengan DASTRPNet:
      B4  → GCN symmetric + LSTM temporal
      Exp → GATv2 attention + cross-modal fusion + CBN + EDL

    Interface identik dengan DASTRPNet.
    """

    def __init__(
        self,
        yolo_weights:  str,
        yolo_conf:     float = 0.25,
        yolo_iou:      float = 0.45,
        gcn_hidden:    int   = GCN_DIM,
        gcn_layers:    int   = 2,
        gcn_dropout:   float = 0.3,
        lstm_dim:      int   = LSTM_DIM,
        lstm_layers:   int   = 1,
        lstm_dropout:  float = 0.2,
        T_agg:         int   = None,
        **kwargs,
    ):
        super().__init__()
        self.T_agg = T_agg

        # ── Backbone (identik dengan DASTRPNet) ───────────────────
        self.yolo = YOLODetector(yolo_weights, yolo_conf, yolo_iou)

        # ── Frame projection ──────────────────────────────────────
        self.frame_proj = nn.Sequential(
            nn.Linear(YOLO_CH, OBJ_DIM), nn.ReLU()
        )

        # ── RoI extractor (sama seperti DASTRPNet) ────────────────
        self.roi_extractor = RoIExtractor(
            in_channels=YOLO_CH, out_dim=OBJ_DIM
        )

        # ── Velocity tracker (sama seperti DASTRPNet) ─────────────
        self._vel_tracker = VelocityTracker()

        # ── Relation encoder ──────────────────────────────────────
        # Input: [boxes_i(4), boxes_j(4), vels_i(2), vels_j(2)] → dim=12
        # Tambah d_ij → 13... sebenarnya d_ij dihitung di dalam,
        # jadi input RelationEncoder = boxes_i + boxes_j + d_ij + dv = 4+4+1+2 = 11
        self.relation_enc = RelationEncoder(
            in_dim  = 11,   # 4(box_i) + 4(box_j) + 1(dist) + 2(Δvel)
            out_dim = RELATION_DIM,
        )

        # ── GCN (Scene-Graph GCN) ─────────────────────────────────
        # Node dim: OBJ_DIM(256) + 4(box) + 2(vel) + 1(motion) = 263
        node_dim = OBJ_DIM + 7
        self.gcn = SceneGCN(
            node_dim     = node_dim,
            relation_dim = RELATION_DIM,
            hidden_dim   = gcn_hidden,
            out_dim      = GCN_DIM,
            n_layers     = gcn_layers,
            dropout      = gcn_dropout,
        )

        # ── Scene embedding projection (setelah graph pooling) ────
        # Concat: [mean_pool, max_pool] → 2 × GCN_DIM
        self.scene_proj = nn.Sequential(
            nn.Linear(2 * GCN_DIM, SCENE_DIM), nn.ReLU(),
            nn.Dropout(0.2),
        )

        # ── LSTM temporal encoder ─────────────────────────────────
        self.lstm = nn.LSTM(
            input_size    = SCENE_DIM,
            hidden_size   = lstm_dim,
            num_layers    = lstm_layers,
            batch_first   = True,
            bidirectional = False,   # Unidirectional sesuai Yu et al. 2021
            dropout       = lstm_dropout if lstm_layers > 1 else 0.0,
        )
        self.lstm_norm = nn.LayerNorm(lstm_dim)

        # ── Risk head ─────────────────────────────────────────────
        self.risk_head = MLPRiskHead(in_dim=lstm_dim)

        # ── Per-frame risk head ───────────────────────────────────
        self.frame_risk_head = nn.Sequential(
            nn.Linear(SCENE_DIM, 64), nn.ReLU(),
            nn.Linear(64, NUM_RISK),
        )

        # ── Dummy domain classifier ───────────────────────────────
        self.domain_clf = nn.Linear(lstm_dim, NUM_DOMAINS)

        print(f"[BaselineB4] Scene-Graph GCN + LSTM")
        print(f"  GCN: hidden={gcn_hidden}, layers={gcn_layers}")
        print(f"  LSTM: dim={lstm_dim}, layers={lstm_layers}")

    def reset_tracker(self):
        self._vel_tracker.reset()

    def _process_frame(
        self,
        images_t:    torch.Tensor,         # (B, 3, H, W)
        gt_boxes_t:  list | None,
        device:      torch.device,
    ) -> torch.Tensor:                     # (B, SCENE_DIM)
        """
        Proses satu frame: YOLO → RoI → GCN → scene embedding.

        Returns scene_emb: (B, SCENE_DIM) — satu embedding per gambar
        """
        B = images_t.size(0)

        # ── Backbone ────────────────────────────────────────────
        feat_map, frame_emb_raw = self.yolo.extract_features(images_t)

        # ── Boxes ───────────────────────────────────────────────
        if gt_boxes_t is not None:
            boxes_list = [b.to(device) for b in gt_boxes_t]
        else:
            boxes_list = self.yolo.detect(images_t)

        # ── Per-image GCN ────────────────────────────────────────
        scene_embs = []
        for i in range(B):
            boxes = boxes_list[i]    # (N_i, 4)
            N     = boxes.size(0)

            if N == 0:
                # Tidak ada deteksi → zero embedding
                scene_embs.append(
                    torch.zeros(SCENE_DIM, device=device)
                )
                continue

            # RoI features dari P4
            roi_feats = self.roi_extractor(
                feat_map[i:i+1], boxes
            )   # (N, OBJ_DIM)

            # Velocity dari tracker
            vels = self._vel_tracker.compute(boxes)   # (N, 2)

            # Motion label dummy (0=gerak untuk semua jika tidak ada GT)
            # Di B4 kita tidak punya motion head output,
            # tapi node feature butuh motion label
            mot_label = torch.zeros(N, dtype=torch.long,
                                    device=device)

            # Scene graph construction
            # Cast ke float32 — GCN forward lebih stabil di fp32
            (node_x, adj_norm, edge_index,
             boxes_i, boxes_j,
             vels_i, vels_j) = build_scene_graph(
                boxes.float(),
                vels.float(),
                mot_label,
                roi_feats.float(),
                device
            )

            # Relation encoding
            if edge_index.size(1) > 0:
                edge_feat = self.relation_enc(
                    boxes_i, boxes_j, vels_i, vels_j
                )   # (E, RELATION_DIM)
            else:
                edge_feat = torch.zeros(0, RELATION_DIM, device=device)

            # GCN forward
            node_out = self.gcn(
                node_x, adj_norm, edge_index, edge_feat
            )   # (N, GCN_DIM)

            # Scene-level pooling: mean + max → (2*GCN_DIM,)
            mean_pool = node_out.mean(dim=0)
            max_pool  = node_out.max(dim=0).values
            pooled    = torch.cat([mean_pool, max_pool], dim=0)

            # Project ke SCENE_DIM
            scene_emb = self.scene_proj(pooled)   # (SCENE_DIM,)
            scene_embs.append(scene_emb)

        return torch.stack(scene_embs, dim=0)   # (B, SCENE_DIM)

    def forward(
        self,
        clip_images:    torch.Tensor,
        domain_ids:     torch.Tensor = None,
        gt_boxes_seq:   list         = None,
        epoch:          int          = 0,
        num_epochs:     int          = 100,
    ) -> dict:
        """
        Forward pass B4.

        Alur:
          Per frame t:
            images_t → YOLO → RoI → GCN → scene_emb_t (B, SCENE_DIM)
          Sequence:
            [scene_emb_0, ..., scene_emb_{T-1}] → LSTM → clip_emb (B, lstm_dim)
          Output:
            clip_emb → MLP Risk Head → risk_pred
        """
        B, T   = clip_images.shape[:2]
        device = clip_images.device

        # Reset velocity tracker di awal setiap clip
        self._vel_tracker.reset()

        # ── Per-frame scene embeddings ────────────────────────────
        scene_embs = []
        for t in range(T):
            images_t   = clip_images[:, t]
            gt_boxes_t = ([gt_boxes_seq[b][t] for b in range(B)]
                          if gt_boxes_seq is not None else None)

            scene_emb_t = self._process_frame(
                images_t, gt_boxes_t, device
            )   # (B, SCENE_DIM)
            scene_embs.append(scene_emb_t)

        # ── LSTM temporal encoding ────────────────────────────────
        # Stack: (B, T, SCENE_DIM)
        scene_seq = torch.stack(scene_embs, dim=1)

        # LSTM forward — unidirectional sesuai Yu et al. 2021
        # output: (B, T, lstm_dim)
        # h_n:    (n_layers, B, lstm_dim)
        lstm_out, (h_n, _) = self.lstm(scene_seq)

        # Clip embedding: final hidden state layer terakhir
        clip_emb = self.lstm_norm(h_n[-1])   # (B, lstm_dim)

        # ── Risk prediction ───────────────────────────────────────
        risk_out = self.risk_head(clip_emb)

        # ── Per-frame risks dari LSTM output ─────────────────────
        frame_logits    = self.frame_risk_head(
            lstm_out.detach()
        )   # (B, T, NUM_RISK)
        all_frame_probs = F.softmax(frame_logits, dim=2)   # (B, T, 3)

        # ── Domain logits (dummy) ─────────────────────────────────
        domain_logits = self.domain_clf(clip_emb.detach())   # (B, 3)

        # ── Dummy per-objek output ────────────────────────────────
        empty_boxes = [torch.zeros(0, 4, device=device) for _ in range(B)]
        empty_mot   = [torch.zeros(0, 2, device=device) for _ in range(B)]
        empty_orisk = [torch.zeros(0, 3, device=device) for _ in range(B)]
        empty_opred = [torch.zeros(0,    device=device,
                                   dtype=torch.long)    for _ in range(B)]

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
