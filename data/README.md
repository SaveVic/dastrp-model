# Data

This folder holds the data loader code (`dataset.py`) and, locally, the
dataset files. **No dataset files are committed**: `imbd-esr/`,
`processed/` and `clips/` are git-ignored.

## Folder layout

```
data/
├── dataset.py                     # DataLoader (code, committed)
│
├── imbd-esr/                      # 1. raw release, downloaded by you
│   ├── annotations/
│   │   ├── split_result.json      #    official train/val/test split
│   │   ├── ugm/ugm_1/meta.json
│   │   ├── ugm/ugm_1/tracks/ugm_1_000042.json
│   │   └── utara/... ringroad/...
│   └── videos/
│       ├── ugm/ugm_1.mp4
│       └── utara/... ringroad/...
│
├── processed/                     # 2. output of tools/convert_imbd_esr.py
│   ├── dastrp/                    #    DASTRP-Net and baselines
│   │   ├── train/ val/ test/
│   │   │   ├── images/  ugm_1__000042.jpg
│   │   │   ├── labels/  ugm_1__000042.txt   (YOLO: class cx cy w h)
│   │   │   ├── tracks/  ugm_1__000042.json  (objects with id, risk, motion, box)
│   │   │   └── meta.json
│   │   ├── class_names.txt
│   │   └── dataset.yaml
│   └── yolo/                      #    same layout, single class "motor"
│
└── clips/                         # 3. optional, output of tools/extract_clips.py
    ├── clips_train.json  clips_val.json  clips_test.json
    ├── clips_summary.json
    └── test/ugm_1_000040_Danger.jpg ...
```

## Steps

```bash
# 1. get IMBD-ESR (https://github.com/SaveVic/dastrp-dataset) into data/imbd-esr

# 2. extract frames + convert annotations
python tools/convert_imbd_esr.py --src data/imbd-esr --dst data/processed/dastrp
python tools/convert_imbd_esr.py --src data/imbd-esr --dst data/processed/yolo --mode yolo

# 3. (optional) list / render the T-frame clips the model is trained on
python tools/extract_clips.py --dataset data/processed/dastrp --out data/clips
python tools/extract_clips.py --dataset data/processed/dastrp --out data/clips \
       --splits test --export jpg --label danger --limit 50
```

`make data data_yolo` runs step 2 with the default paths.

## Preprocessing details

| Step | Rule | Where |
|---|---|---|
| Frame selection, train | annotated frames ± 16 frames, plus 10 % of the remaining frames (seed 42) | `convert_imbd_esr.py --margin --bg-ratio --seed` |
| Frame selection, val/test | all frames | `convert_imbd_esr.py` |
| Box format | `[cx, cy, w, h]` normalised to [0, 1] and clipped | `convert_imbd_esr.py` |
| Frame risk | max of per-object risk in the frame (0 if no object) | `dataset.aggregate_frame_risk` |
| Clip | *T* = 4 consecutive extracted frames of one video | `dataset.ClipSampler` |
| Clip stride | train 2 (50 % overlap); val/test 4 (non-overlapping) | `dataset.get_dataloaders` |
| Clip label | max frame risk in the clip | `scripts/train.py` |
| Image input | resized to 640 × 640, ImageNet normalisation | `dataset._load_image` |
| Augmentation (train) | brightness jitter, horizontal flip (same for all frames of a clip) | `dataset._load_image` |
| Domain | from the video name prefix: `ugm` = 0, `utara` = 1, `ringroad` = 2 | `dataset.infer_domain` |

`tools/split_videos.py` produces the stratified split (`split_result.json`)
from per-video statistics, and `tools/prepare_dataset.py` is the internal
pipeline that builds the same processed layout from the raw CVAT exports
(not public). Both are included for transparency; with the public release
only `convert_imbd_esr.py` is needed.
