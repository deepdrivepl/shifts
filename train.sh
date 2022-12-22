#!/bin/bash

# TRAIN_FLAIR="C:/Users/karol/Downloads/shifts_ms_pt1/msseg/train/flair"
# TRAIN_GT="C:/Users/karol/Downloads/shifts_ms_pt1/msseg/train/gt"
# VAL_FLAIR="C:/Users/karol/Downloads/shifts_ms_pt1/msseg/eval_in/flair"
# VAL_GT="C:/Users/karol/Downloads/shifts_ms_pt1/msseg/eval_in/gt"
TRAIN_FLAIR="/mnt/12TB/projects/shifts2022/data/shifts_ms_pt1/msseg/train/flair"
TRAIN_GT="/mnt/12TB/projects/shifts2022/data/shifts_ms_pt1/msseg/train/gt"
VAL_FLAIR="/mnt/12TB/projects/shifts2022/data/shifts_ms_pt1/msseg/eval_in/flair"
VAL_GT="/mnt/12TB/projects/shifts2022/data/shifts_ms_pt1/msseg/eval_in/gt"


for seed in 42
do
	python mswml/train.py \
	--seed $seed \
	--path_train_data $TRAIN_FLAIR \
	--path_train_gts  $TRAIN_GT \
	--path_val_data   $VAL_FLAIR \
	--path_val_gts    $VAL_GT \
	--path_save "trained_models/seed${seed}" \
	--exp_name "test"
done
