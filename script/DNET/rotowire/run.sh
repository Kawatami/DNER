#!/bin/bash

for N_RUN in 0 1 2
do

  echo ====================== VANILLA RUN $N_RUN =============================

  python ../run.py \
  --task SpanClassificationTask \
  --dataset RotoWireDataModule \
  --data_tag roto_DNET \
  --model BertLinear \
  --loss BCELoss \
  --metrics Accuracy-Accuracy test-CNETClassificationReport \
  --default_root_dir $LOG_ROOT/task_2/rotowire/coling/vanilla \
  --data_directory $DATA_ROOT/CNER_datasets/CNER_RotoWire \
  --batch_size 4 \
  --gpus 1 \
  --callbacks SpanClassificationDataCollector \
  --collector_log_dir $INFERENCE_ROOT/task_1/rotowire/coling/vanilla  \
  --single_gpu_testing \
  --test_files      seen_unseen.json unseen.json seen.json all.json  \
  --test_set_names seen_unseen       unseen      seen      all \
  --gradient_clip_val 3

done