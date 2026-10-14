"""
data/dataset.py
===============
DataLoader untuk DASTRP-Net — berbasis clip (sequence-aware batching).

Struktur dataset yang diharapkan:
  dataset/
    train/
      images/   ringroad_1__000001.jpg  (1280×720)
      labels/   ringroad_1__000001.txt  (YOLO: class_id cx cy w h)
      tracks/   ringroad_1__000001.json (id, risk, label per objek)
      meta.json
    val/  (idem, semua frame dipakai)
    test/ (idem, semua frame dipakai)
    class_names.txt
    dataset.yaml

Catatan penting:
  - File .txt (YOLO label) TIDAK dibaca di sini — hanya dipakai saat
    training YOLO detektor. Semua label dibaca dari tracks/*.json.
  - Domain di-infer dari prefix nama file: ugm_* / utara_* / ringroad_*.
  - Setiap sample adalah clip T frame berurutan dari satu video.
    Shuffle dilakukan antar clip, bukan antar frame, sehingga urutan
    temporal dalam clip selalu terjaga untuk velocity tracker dan GNN.

Output satu sample (dict):
  images      : (T, 3, H, W)   frame berurutan dalam clip
  boxes_seq   : list[T] tensor (N_t, 4) [cx,cy,w,h] normalized
  labels_seq  : list[T] tensor (N_t,)   0=gerak, 1=diam
  risk_seq    : list[T] tensor (N_t,)   0=safe, 1=warning, 2=danger per obj
  track_seq   : list[T] tensor (N_t,)   object ID dari anotator
  frame_risks : (T,)  risk level agregat per frame dalam clip
  domain_id   : ()    0=UGM, 1=Utara, 2=Ringroad
  video_name  : str
  clip_stems  : list[T] str — nama stem tiap frame dalam clip
"""

import cv2
import json
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ─────────────────────────────────────────────
# Konstanta global
# ─────────────────────────────────────────────

DOMAIN_MAP = {"ugm": 0, "utara": 1, "ringroad": 2}
RISK_MAP   = {"safe": 0, "warning": 1, "danger": 2}
LABEL_MAP  = {"gerak": 0, "diam": 1}

DOMAIN_NAMES = {0: "UGM", 1: "Utara", 2: "Ringroad"}
RISK_NAMES   = {0: "Safe", 1: "Warning", 2: "Danger"}
LABEL_NAMES  = {0: "gerak", 1: "diam"}


# ─────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────

def infer_domain(name: str) -> int:
    """
    Infer domain ID dari nama file atau nama video.
    Contoh: 'ringroad_1__000013' → 2
            'ugm_4'              → 0
    """
    stem = Path(name).stem.lower()
    for prefix, did in DOMAIN_MAP.items():
        if stem.startswith(prefix):
            return did
    return -1   # unknown


def parse_stem(stem: str) -> tuple:
    """
    'ringroad_1__000013' → ('ringroad_1', 13)
    Digunakan untuk mendapatkan nama video dan frame index.
    """
    parts = stem.rsplit("__", 1)
    if len(parts) == 2:
        try:
            return parts[0], int(parts[1])
        except ValueError:
            pass
    return stem, 0


def aggregate_frame_risk(objects: list) -> int:
    """
    Risk level frame = max risk dari semua objek di frame tersebut.
      - Semua safe / tidak ada objek → 0 (safe)
      - Ada warning, tidak ada danger → 1 (warning)
      - Ada danger → 2 (danger)
    """
    if not objects:
        return 0
    return max(int(o.get("risk_id", 0)) for o in objects)


# ─────────────────────────────────────────────
# ClipSampler
# ─────────────────────────────────────────────

class ClipSampler:
    """
    Membangun daftar semua clip dari satu video.

    Clip = T frame berurutan. Digeser sebesar `stride` frame
    per langkah (stride < T → clip saling overlap).

    Args:
        stems  : list stem file yang sudah diurutkan temporal
                 (urutan dijamin karena format 6-digit zero-padded)
        T      : panjang clip (jumlah frame per clip)
        stride : langkah antar clip awal
                 - stride = T   → tidak ada overlap (non-overlapping)
                 - stride = T/2 → overlap 50% (rekomendasi)
                 - stride = 1   → sliding window penuh (lebih banyak clip)
    """

    def __init__(self, stems: list, T: int = 4, stride: int = 2):
        self.stems  = stems
        self.T      = T
        self.stride = stride
        self.clips  = []   # list of list[stem], panjang T

        n = len(stems)
        for start in range(0, n - T + 1, stride):
            self.clips.append(stems[start : start + T])

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx: int) -> list:
        return self.clips[idx]


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class DASTRPClipDataset(Dataset):
    """
    Dataset berbasis clip untuk DASTRP-Net.

    Setiap sample = 1 clip (T frame berurutan dari 1 video).
    Shuffle di DataLoader dilakukan antar clip, sehingga urutan
    temporal dalam clip selalu terjaga.

    Args:
        split_dir : path ke folder split, e.g. 'dataset/train'
        T         : panjang clip dalam jumlah frame
        stride    : langkah antar clip (rekomendasi: T//2)
        img_size  : ukuran resize gambar (default: 640)
        augment   : aktifkan augmentasi (gunakan hanya untuk split train)
    """

    def __init__(
        self,
        split_dir: str,
        T:         int  = 4,
        stride:    int  = 2,
        img_size:  int  = 640,
        augment:   bool = False,
    ):
        self.split_dir = Path(split_dir)
        self.img_dir   = self.split_dir / "images"
        self.trk_dir   = self.split_dir / "tracks"
        self.T         = T
        self.stride    = stride
        self.img_size  = img_size
        self.augment   = augment

        assert self.img_dir.exists(), \
            f"Folder images tidak ditemukan: {self.img_dir}"
        assert self.trk_dir.exists(), \
            f"Folder tracks tidak ditemukan: {self.trk_dir}"

        # ── Kelompokkan stem per video, urutkan temporal ──────
        # Kunci: nama video (prefix sebelum '__')
        # Nilai: list stem yang punya pasangan track JSON
        video_stems: dict = defaultdict(list)
        for img_path in sorted(self.img_dir.glob("*.jpg")):
            stem = img_path.stem
            trk_path = self.trk_dir / f"{stem}.json"
            if not trk_path.exists():
                continue   # skip jika tidak ada track
            video_name, _ = parse_stem(stem)
            video_stems[video_name].append(stem)

        assert video_stems, \
            f"Tidak ada frame dengan pasangan track di {self.img_dir}"

        # ── Cache track JSON ke RAM (kecil, ~MB) ──────────────
        # Menghindari disk I/O berulang di setiap __getitem__
        self._track_cache: dict = {}
        for stems in video_stems.values():
            for stem in stems:
                with open(self.trk_dir / f"{stem}.json") as f:
                    self._track_cache[stem] = json.load(f)

        # ── Bangun semua clip dari semua video ─────────────────
        # all_clips: list of (video_name, list_of_T_stems)
        self.all_clips: list = []
        self._video_clip_count: dict = {}   # untuk diagnostik

        for video_name, stems in sorted(video_stems.items()):
            # stems sudah terurut karena glob + sort alphabetical
            # dan format 6-digit zero-padded menjamin urutan temporal
            sampler = ClipSampler(stems, T=T, stride=stride)
            n_clips = len(sampler)
            self._video_clip_count[video_name] = n_clips
            for ci in range(n_clips):
                self.all_clips.append((video_name, sampler[ci]))

        # ── Transform image ───────────────────────────────────
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std= [0.229, 0.224, 0.225],
            ),
        ])

        # ── Print ringkasan ───────────────────────────────────
        self._print_summary()

    # ── Internal: ringkasan dataset ───────────────────────────

    def _print_summary(self):
        split_name = self.split_dir.name.upper()
        n_videos   = len(self._video_clip_count)
        n_clips    = len(self.all_clips)

        print(f"[{split_name}] {n_clips} clip "
              f"(T={self.T}, stride={self.stride}) "
              f"dari {n_videos} video")

        # Hitung distribusi domain dan risk (sampel 500 clip pertama
        # agar tidak terlalu lambat saat dataset besar)
        domain_cnt = defaultdict(int)
        risk_cnt   = defaultdict(int)
        empty_cnt  = 0
        sample_limit = min(500, n_clips)

        for i in range(sample_limit):
            video_name, clip_stems = self.all_clips[i]
            domain_cnt[DOMAIN_NAMES.get(infer_domain(video_name), "?")] += 1
            # Ambil frame tengah clip sebagai representasi
            mid_stem = clip_stems[self.T // 2]
            trk_path = self.trk_dir / f"{mid_stem}.json"
            with open(trk_path) as f:
                trk = json.load(f)
            objects = trk.get("objects", [])
            if not objects:
                empty_cnt += 1
            else:
                fr = aggregate_frame_risk(objects)
                risk_cnt[RISK_NAMES[fr]] += 1

        print(f"  Domain (sampel {sample_limit}): {dict(domain_cnt)}")
        print(f"  Risk mid-frame: {dict(risk_cnt)} | Empty: {empty_cnt}")
        print(f"  Clip per video: "
              + ", ".join(f"{v}={n}" for v, n
                          in sorted(self._video_clip_count.items())))

    # ── Internal: load satu frame ─────────────────────────────

    def _load_image(self, stem: str, flip: bool = False) -> torch.Tensor:
        """Load, resize, augment, dan transform satu frame."""
        img_path = self.img_dir / f"{stem}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image tidak ditemukan: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        if self.augment:
            # Brightness jitter
            factor = np.random.uniform(0.75, 1.25)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
            # Gaussian blur (simulasi motion blur / blur Ringroad)
            if np.random.rand() > 0.7:
                ksize = int(np.random.choice([3, 5]))
                img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        if flip:
            img = cv2.flip(img, 1)

        return self.transform(img)   # (3, H, W)

    # ── Internal: load track satu frame ───────────────────────

    def _load_track(self, stem: str, flip: bool = False) -> tuple:
        """
        Load tracks/*.json untuk satu frame.

        Returns:
            boxes      : (N, 4) float32  [cx, cy, w, h] normalized
            labels     : (N,)   int64    0=gerak, 1=diam
            risk_ids   : (N,)   int64    0=safe, 1=warning, 2=danger
            track_ids  : (N,)   int64    object ID dari anotator (-1 jika null)
            frame_risk : int             risk agregat frame
        """
        trk = self._track_cache.get(stem)
        if trk is None:
            with open(self.trk_dir / f"{stem}.json") as f:
                trk = json.load(f)
        objects = trk.get("objects", [])

        N = len(objects)
        boxes     = np.zeros((N, 4), dtype=np.float32)
        labels    = np.zeros(N,      dtype=np.int64)
        risk_ids  = np.zeros(N,      dtype=np.int64)
        track_ids = np.full(N, -1,   dtype=np.int64)

        for i, obj in enumerate(objects):
            cx = float(obj["cx"])
            cy = float(obj["cy"])
            w  = float(obj["w"])
            h  = float(obj["h"])

            if flip:
                cx = 1.0 - cx   # flip horizontal: x_center jadi 1-x

            boxes[i] = [cx, cy, w, h]

            # Label gerak/diam — baca class_id langsung atau fallback dari string
            raw_label = obj.get("label", "gerak")
            labels[i] = int(obj.get("class_id",
                            LABEL_MAP.get(str(raw_label).lower(), 0)))

            # Risk level — baca risk_id langsung atau fallback dari string
            raw_risk = obj.get("risk", "safe")
            risk_ids[i] = int(obj.get("risk_id",
                              RISK_MAP.get(str(raw_risk).lower(), 0)))

            # Object ID untuk tracking evaluation
            tid = obj.get("id")
            track_ids[i] = int(tid) if tid is not None else -1

        frame_risk = aggregate_frame_risk(objects)

        return (
            torch.from_numpy(boxes),       # (N, 4)
            torch.from_numpy(labels),      # (N,)
            torch.from_numpy(risk_ids),    # (N,)
            torch.from_numpy(track_ids),   # (N,)
            frame_risk,                    # int
        )

    # ── Dataset interface ──────────────────────────────────────

    def __len__(self) -> int:
        return len(self.all_clips)

    def __getitem__(self, idx: int) -> dict:
        video_name, clip_stems = self.all_clips[idx]
        domain_id = infer_domain(video_name)

        # Satu keputusan flip per clip (konsisten untuk semua frame dalam clip)
        flip = self.augment and (np.random.rand() > 0.5)

        # ── Load T frame berurutan ────────────────────────────
        images      = []
        boxes_seq   = []
        labels_seq  = []
        risk_seq    = []
        track_seq   = []
        frame_risks = []

        for stem in clip_stems:
            # Image
            images.append(self._load_image(stem, flip))

            # Track
            boxes, labels, risk_ids, track_ids, frame_risk = \
                self._load_track(stem, flip)

            boxes_seq.append(boxes)
            labels_seq.append(labels)
            risk_seq.append(risk_ids)
            track_seq.append(track_ids)
            frame_risks.append(frame_risk)

        return {
            # Tensor yang bisa di-stack antar sample dalam batch
            "images":      torch.stack(images),                         # (T, 3, H, W)
            "frame_risks": torch.tensor(frame_risks, dtype=torch.long), # (T,)
            "domain_id":   torch.tensor(domain_id,   dtype=torch.long), # scalar

            # List of tensor — N berbeda tiap frame, tidak bisa di-stack
            "boxes_seq":   boxes_seq,    # list[T] → each (N_t, 4)
            "labels_seq":  labels_seq,   # list[T] → each (N_t,)
            "risk_seq":    risk_seq,     # list[T] → each (N_t,)
            "track_seq":   track_seq,    # list[T] → each (N_t,)

            # Metadata
            "video_name":  video_name,   # str
            "clip_stems":  clip_stems,   # list[T] str
        }


# ─────────────────────────────────────────────
# Collate function
# ─────────────────────────────────────────────

def clip_collate_fn(batch: list) -> dict:
    """
    Collate untuk DASTRPClipDataset.

    Tensor yang shape-nya seragam antar sample → di-stack menjadi batch tensor.
    Tensor dengan N objek berbeda tiap frame → tetap list of list of tensor.

    Struktur output:
      images       : (B, T, 3, H, W)
      frame_risks  : (B, T)
      domain_ids   : (B,)
      boxes_seq    : list[B] of list[T] of tensor(N_t, 4)
      labels_seq   : list[B] of list[T] of tensor(N_t,)
      risk_seq     : list[B] of list[T] of tensor(N_t,)
      track_seq    : list[B] of list[T] of tensor(N_t,)
      video_names  : list[B] str
      clip_stems   : list[B] of list[T] str
    """
    return {
        "images":      torch.stack([b["images"]      for b in batch]),  # (B,T,3,H,W)
        "frame_risks": torch.stack([b["frame_risks"] for b in batch]),  # (B,T)
        "domain_ids":  torch.stack([b["domain_id"]   for b in batch]),  # (B,)
        "boxes_seq":   [b["boxes_seq"]  for b in batch],
        "labels_seq":  [b["labels_seq"] for b in batch],
        "risk_seq":    [b["risk_seq"]   for b in batch],
        "track_seq":   [b["track_seq"]  for b in batch],
        "video_names": [b["video_name"] for b in batch],
        "clip_stems":  [b["clip_stems"] for b in batch],
    }


# ─────────────────────────────────────────────
# Factory: buat semua DataLoader sekaligus
# ─────────────────────────────────────────────

def seed_worker(worker_id: int):
    """Seed python/numpy RNG in each DataLoader worker (reproducibility)."""
    import random
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_dataloaders(
    dataset_root: str,
    T:            int   = 4,
    stride:       int   = 2,
    img_size:     int   = 640,
    batch_size:   int   = 8,
    num_workers:  int   = 4,
    seed:         int   = 42,
    splits:       list  = None,
) -> dict:
    """
    Buat DataLoader untuk train / val / test.

    Args:
        dataset_root : root folder dataset (berisi train/, val/, test/)
        T            : panjang clip dalam frame
        stride       : langkah antar clip (T//2 = overlap 50%)
        img_size     : ukuran resize gambar
        batch_size   : jumlah clip per batch
        num_workers  : jumlah worker DataLoader
        seed         : seed dasar; tiap split memakai generator terpisah
                       (train=seed, val=seed+1, test=seed+2)
        splits       : subset split yang dibuat (default: semua)

    Returns:
        dict dengan key 'train', 'val', 'test' (hanya split yang ada)

    Catatan:
        - train  : shuffle=True (antar clip), augment=True, drop_last=True
        - val    : shuffle=False, augment=False, drop_last=False
        - test   : shuffle=False, augment=False, drop_last=False
        - val/test menggunakan stride=T agar setiap frame hanya muncul
          sekali dalam evaluasi (tidak overlap antar clip)
    """
    root    = Path(dataset_root)
    loaders = {}
    split_seeds = {"train": seed, "val": seed + 1, "test": seed + 2}

    for split in (splits or ["train", "val", "test"]):
        split_dir = root / split
        if not (split_dir / "images").exists():
            print(f"[SKIP] '{split}' tidak ditemukan di {split_dir}")
            continue

        is_train = (split == "train")

        # Val/test: stride=T agar tidak ada frame yang dihitung 2x
        effective_stride = stride if is_train else T

        ds = DASTRPClipDataset(
            split_dir = str(split_dir),
            T         = T,
            stride    = effective_stride,
            img_size  = img_size,
            augment   = is_train,
        )

        # Generator independen per split
        g = torch.Generator()
        g.manual_seed(split_seeds[split])

        workers = num_workers if is_train else min(2, num_workers)
        loaders[split] = DataLoader(
            ds,
            batch_size         = batch_size,
            shuffle            = is_train,
            num_workers        = workers,
            collate_fn         = clip_collate_fn,
            pin_memory         = torch.cuda.is_available(),
            drop_last          = is_train,
            persistent_workers = (workers > 0),
            worker_init_fn     = seed_worker,
            generator          = g,
            prefetch_factor    = (1 if workers > 0 else None),
        )

    return loaders


# ─────────────────────────────────────────────
# Quick test (jalankan langsung: python dataset.py)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        help="Root dataset (berisi train/val/test)")
    parser.add_argument("--T",      type=int, default=4)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch",  type=int, default=4)
    args = parser.parse_args()

    loaders = get_dataloaders(
        dataset_root = args.dataset,
        T            = args.T,
        stride       = args.stride,
        batch_size   = args.batch,
        num_workers  = 0,   # 0 untuk debug
    )

    for split, loader in loaders.items():
        print(f"\n=== {split.upper()} ===")
        batch = next(iter(loader))

        B, T_  = batch["images"].shape[:2]
        print(f"  images      : {tuple(batch['images'].shape)}")
        print(f"  frame_risks : {tuple(batch['frame_risks'].shape)}")
        print(f"  domain_ids  : {batch['domain_ids'].tolist()}")
        print(f"  video_names : {batch['video_names']}")

        # Cek N objek tiap frame tiap sample
        for b in range(min(B, 2)):
            Ns = [batch["boxes_seq"][b][t].shape[0] for t in range(T_)]
            print(f"  sample[{b}] N per frame: {Ns}")

        print(f"  ✓ batch OK")
