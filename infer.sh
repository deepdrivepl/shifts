#!/bin/bash

MODEL_PATH="runs/loss100/xunet-loss-ndsc-lr/version_0/models/epoch=70-7810-dice_loss=0.33234.ckpt"
PARAMS_PATH="params/xunet-loss-ndsc-lr.py"
PLOT_TITLE="xunet-loss-ndsc-lr-crop"
SAVE_DIR="xunet-loss-ndsc-lr-crop"

N_JOBS=1
THRESHOLD=0.35
TH_STEP=5
UNCERTAINTY='entropy_of_expected'


# === TEST METRICS ===
# python mswml/test.py --path_model $MODEL_PATH \
#                      --path_data "data/shifts_ms_pt1/msseg/dev_in/flair" "data/shifts_ms_pt2/best/dev_in/flair" \
#                      --path_gts "data/shifts_ms_pt1/msseg/dev_in/gt" "data/shifts_ms_pt2/best/dev_in/gt" \
#                      --path_bm "data/shifts_ms_pt1/msseg/dev_in/fg_mask" "data/shifts_ms_pt2/best/dev_in/fg_mask" \
#                      --path_save "test_predictions/$SAVE_DIR/dev_in" \
#                      --n_jobs $N_JOBS \
#                      --threshold $THRESHOLD \
#                      --uncertainty $UNCERTAINTY \
#                      --path_params $PARAMS_PATH

# python mswml/test.py --path_model $MODEL_PATH \
#                      --path_data "data/shifts_ms_pt1/msseg/eval_in/flair" "data/shifts_ms_pt2/best/eval_in/flair" \
#                      --path_gts "data/shifts_ms_pt1/msseg/eval_in/gt" "data/shifts_ms_pt2/best/eval_in/gt" \
#                      --path_bm "data/shifts_ms_pt1/msseg/eval_in/fg_mask" "data/shifts_ms_pt2/best/eval_in/fg_mask" \
#                      --path_save "test_predictions/$SAVE_DIR/eval_in" \
#                      --n_jobs $N_JOBS \
#                      --threshold $THRESHOLD \
#                      --uncertainty $UNCERTAINTY \
#                      --path_params $PARAMS_PATH

# python mswml/test.py --path_model $MODEL_PATH \
#                      --path_data 'data/shifts_ms_pt2/ljubljana/dev_out/flair' \
#                      --path_gts 'data/shifts_ms_pt2/ljubljana/dev_out/gt' \
#                      --path_bm 'data/shifts_ms_pt2/ljubljana/dev_out/fg_mask' \
#                      --path_save "test_predictions/$SAVE_DIR/dev_out" \
#                      --n_jobs $N_JOBS \
#                      --threshold $THRESHOLD \
#                      --uncertainty $UNCERTAINTY \
#                      --path_params $PARAMS_PATH


# === THRESHOLD ADJUSTMENT ===
# python mswml/threshold_adjustment.py --path_model $MODEL_PATH \
#                                      --path_data "data/shifts_ms_pt1/msseg/dev_in/flair" "data/shifts_ms_pt2/best/dev_in/flair" \
#                                      --path_gts "data/shifts_ms_pt1/msseg/dev_in/gt" "data/shifts_ms_pt2/best/dev_in/gt" \
#                                      --path_bm "data/shifts_ms_pt1/msseg/dev_in/fg_mask" "data/shifts_ms_pt2/best/dev_in/fg_mask" \
#                                      --path_save "test_predictions/$SAVE_DIR/dev_in" \
#                                      --n_jobs $N_JOBS \
#                                      --plot_title "$PLOT_TITLE - dev_in" \
#                                      --th_step $TH_STEP \
#                                      --path_params $PARAMS_PATH

# python mswml/threshold_adjustment.py --path_model $MODEL_PATH \
#                                      --path_data "data/shifts_ms_pt1/msseg/eval_in/flair" "data/shifts_ms_pt2/best/eval_in/flair" \
#                                      --path_gts "data/shifts_ms_pt1/msseg/eval_in/gt" "data/shifts_ms_pt2/best/eval_in/gt" \
#                                      --path_bm "data/shifts_ms_pt1/msseg/eval_in/fg_mask" "data/shifts_ms_pt2/best/eval_in/fg_mask" \
#                                      --path_save "test_predictions/$SAVE_DIR/eval_in" \
#                                      --n_jobs $N_JOBS \
#                                      --plot_title "$PLOT_TITLE - eval_in" \
#                                      --th_step $TH_STEP \
#                                      --path_params $PARAMS_PATH

# python mswml/threshold_adjustment.py --path_model $MODEL_PATH \
#                                      --path_data 'data/shifts_ms_pt2/ljubljana/dev_out/flair' \
#                                      --path_gts 'data/shifts_ms_pt2/ljubljana/dev_out/gt' \
#                                      --path_bm 'data/shifts_ms_pt2/ljubljana/dev_out/fg_mask' \
#                                      --path_save "test_predictions/$SAVE_DIR/dev_out" \
#                                      --n_jobs $N_JOBS \
#                                      --plot_title "$PLOT_TITLE - dev_out" \
#                                      --th_step $TH_STEP \
#                                      --path_params $PARAMS_PATH


# === RETENTION CURVES ===
# python mswml/retention_curves.py --path_model $MODEL_PATH \
#                                  --path_data "data/shifts_ms_pt1/msseg/dev_in/flair" "data/shifts_ms_pt2/best/dev_in/flair" \
#                                  --path_gts "data/shifts_ms_pt1/msseg/dev_in/gt" "data/shifts_ms_pt2/best/dev_in/gt" \
#                                  --path_bm "data/shifts_ms_pt1/msseg/dev_in/fg_mask" "data/shifts_ms_pt2/best/dev_in/fg_mask" \
#                                  --path_save "test_predictions/$SAVE_DIR/dev_in" \
#                                  --n_jobs $N_JOBS \
#                                  --plot_title "$PLOT_TITLE - dev_in" \
#                                  --threshold $THRESHOLD \
#                                  --path_params $PARAMS_PATH

# python mswml/retention_curves.py --path_model $MODEL_PATH \
#                                  --path_data "data/shifts_ms_pt1/msseg/eval_in/flair" "data/shifts_ms_pt2/best/eval_in/flair" \
#                                  --path_gts "data/shifts_ms_pt1/msseg/eval_in/gt" "data/shifts_ms_pt2/best/eval_in/gt" \
#                                  --path_bm "data/shifts_ms_pt1/msseg/eval_in/fg_mask" "data/shifts_ms_pt2/best/eval_in/fg_mask" \
#                                  --path_save "test_predictions/$SAVE_DIR/eval_in" \
#                                  --n_jobs $N_JOBS \
#                                  --plot_title "$PLOT_TITLE - eval_in" \
#                                  --threshold $THRESHOLD \
#                                  --path_params $PARAMS_PATH

# python mswml/retention_curves.py --path_model $MODEL_PATH \
#                                  --path_data 'data/shifts_ms_pt2/ljubljana/dev_out/flair' \
#                                  --path_gts 'data/shifts_ms_pt2/ljubljana/dev_out/gt' \
#                                  --path_bm 'data/shifts_ms_pt2/ljubljana/dev_out/fg_mask' \
#                                  --path_save "test_predictions/$SAVE_DIR/dev_out" \
#                                  --n_jobs $N_JOBS \
#                                  --plot_title "$PLOT_TITLE - dev_out" \
#                                  --threshold $THRESHOLD \
#                                  --path_params $PARAMS_PATH


# === INFERENCE - SAVE PREDICTIONS ===
# python mswml/inference.py --path_pred "test_predictions/$SAVE_DIR/dev_in/predictions" \
#                           --path_model $MODEL_PATH \
#                           --path_data "data/shifts_ms_pt1/msseg/dev_in/flair" "data/shifts_ms_pt2/best/dev_in/flair" \
#                           --path_bm "data/shifts_ms_pt1/msseg/dev_in/fg_mask" "data/shifts_ms_pt2/best/dev_in/fg_mask" \
#                           --threshold $THRESHOLD \
#                           --uncertainty $UNCERTAINTY \
#                           --path_params $PARAMS_PATH

# python mswml/inference.py --path_pred "test_predictions/$SAVE_DIR/eval_in/predictions" \
#                           --path_model $MODEL_PATH \
#                           --path_data "data/shifts_ms_pt1/msseg/eval_in/flair" "data/shifts_ms_pt2/best/eval_in/flair" \
#                           --path_bm "data/shifts_ms_pt1/msseg/eval_in/fg_mask" "data/shifts_ms_pt2/best/eval_in/fg_mask" \
#                           --threshold $THRESHOLD \
#                           --uncertainty $UNCERTAINTY \
#                           --path_params $PARAMS_PATH

# python mswml/inference.py --path_pred "test_predictions/$SAVE_DIR/dev_out/predictions" \
#                           --path_model $MODEL_PATH \
#                           --path_data 'data/shifts_ms_pt2/ljubljana/dev_out/flair' \
#                           --path_bm 'data/shifts_ms_pt2/ljubljana/dev_out/fg_mask' \
#                           --threshold $THRESHOLD \
#                           --uncertainty $UNCERTAINTY \
#                           --path_params $PARAMS_PATH