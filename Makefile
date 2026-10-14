# =================================================================
#  DASTRP-Net — full reproduction pipeline
#  Override any path on the command line, e.g.
#     make train DATA_SRC=../imbd-esr RESULTS=../runs
# =================================================================

# DATA_SRC  : public IMBD-ESR release (contains annotations/ and videos/)
# DATASET   : processed dataset for DASTRP-Net (output of `make data`)
# YOLO_DATA : processed dataset for the detector (output of `make data_yolo`)
PYTHON     ?= python
DATA_SRC   ?= data/imbd-esr
DATASET    ?= data/processed/dastrp
YOLO_DATA  ?= data/processed/yolo
RESULTS    ?= runs
YOLO       ?= $(RESULTS)/yolo/v11n/weights/best.pt
WORKERS    ?= 8
SEED       ?= 42

TRAIN_ARGS = --dataset "$(DATASET)" --yolo "$(YOLO)" --workers $(WORKERS) --seed $(SEED)

# ── 1. Data ───────────────────────────────────────────────────────
data:
	$(PYTHON) tools/convert_imbd_esr.py --src "$(DATA_SRC)" --dst "$(DATASET)"

data_yolo:
	$(PYTHON) tools/convert_imbd_esr.py --src "$(DATA_SRC)" --dst "$(YOLO_DATA)" --mode yolo

# ── 2. Detector (YOLO11n, single class "motor") ───────────────────
#   The paper compares v8n / v8s / 11n / 11s and selects YOLO11n.
YOLO_EPOCHS ?= 100
YOLO_IMGSZ  ?= 640
YOLO_MODEL_v8n  = yolov8n.pt
YOLO_MODEL_v8s  = yolov8s.pt
YOLO_MODEL_v11n = yolo11n.pt
YOLO_MODEL_v11s = yolo11s.pt
yolo_%:
	yolo detect train data="$(YOLO_DATA)/dataset.yaml" model=$(YOLO_MODEL_$*) \
	    epochs=$(YOLO_EPOCHS) imgsz=$(YOLO_IMGSZ) seed=$(SEED) \
	    project="$(RESULTS)/yolo" name=$* exist_ok=True

yolo: yolo_v11n
yolo_all: yolo_v8n yolo_v8s yolo_v11n yolo_v11s

# ── 3. DASTRP-Net (Exp3: full model, stratified split) ────────────
train:
	$(PYTHON) scripts/train.py $(TRAIN_ARGS) --epochs 100 --patience 20 \
	    --save-dir "$(RESULTS)/exp3"

# ── 4. Ablation study (Table: A1–A6) ──────────────────────────────
ABLATIONS = no_gnn no_cbn no_fusion no_edl no_velocity no_obj_risk
ablation_%:
	$(PYTHON) scripts/train.py $(TRAIN_ARGS) --ablation $* \
	    --epochs 50 --patience 15 --save-dir "$(RESULTS)/ablation/$*"

ablation: $(addprefix ablation_,$(ABLATIONS))

# ── 5. Baselines (Table: B1–B3; B4–B5 are additional) ─────────────
baseline_%:
	$(PYTHON) scripts/train_baseline.py $(TRAIN_ARGS) --model $* \
	    --epochs 50 --patience 15 --save-dir "$(RESULTS)/baseline/$*"

baselines: baseline_b1 baseline_b2 baseline_b3
baselines_all: baselines baseline_b4 baseline_b5

# ── 6. Evaluation on the test split ───────────────────────────────
eval:
	$(PYTHON) scripts/evaluate.py --config "$(RESULTS)/exp3/config.json" --split test

eval_ablation:
	$(PYTHON) scripts/evaluate_ablation_batch.py --results-dir "$(RESULTS)" --skip-existing

eval_baseline_%:
	$(PYTHON) scripts/evaluate.py --config "$(RESULTS)/baseline/$*/config.json" --split test

eval_baselines: eval_baseline_b1 eval_baseline_b2 eval_baseline_b3

# ── 7. Tables & figures ───────────────────────────────────────────
report:
	$(PYTHON) scripts/extract_ablation_results.py --results-dir "$(RESULTS)" --out-dir "$(RESULTS)/ablation_report"
	$(PYTHON) scripts/extract_ablation_data.py    --results-dir "$(RESULTS)" --out "$(RESULTS)/ablation_data.json"
	$(PYTHON) scripts/extract_sota_data.py        --results-dir "$(RESULTS)" --out "$(RESULTS)/sota_data.json"

figures:
	$(PYTHON) scripts/plot_yolo_curves.py        --results-dir "$(RESULTS)/yolo" --out-dir "$(RESULTS)/figures"
	$(PYTHON) scripts/plot_result_figures.py     --exp2-dir "$(RESULTS)/exp2" --exp3-dir "$(RESULTS)/exp3" --out-dir "$(RESULTS)/figures"
	$(PYTHON) scripts/plot_uncertainty_figure.py --exp3-dir "$(RESULTS)/exp3" --out-dir "$(RESULTS)/figures"
	$(PYTHON) scripts/generate_qualitative_figure.py --config "$(RESULTS)/exp3/config.json" --out-dir "$(RESULTS)/figures/qualitative"

# ── Everything ────────────────────────────────────────────────────
all: data data_yolo yolo train eval ablation eval_ablation baselines eval_baselines report

.PHONY: data data_yolo yolo yolo_all train ablation baselines baselines_all \
        eval eval_ablation eval_baselines report figures all

# ── Utilities ─────────────────────────────────────────────────────
clips:
	$(PYTHON) tools/extract_clips.py --dataset "$(DATASET)" --out data/clips

check_env:
	$(PYTHON) tools/check_env.py --forward

weights:
	$(PYTHON) tools/download_weights.py

export_weights:
	$(PYTHON) tools/export_checkpoint.py --yolo "$(YOLO)" --run "$(RESULTS)/exp3" --name dastrp_exp3 --out weights

check:
	$(PYTHON) tools/check_release.py --weights weights

.PHONY: clips check_env weights export_weights check
