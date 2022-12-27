#!/bin/bash

for N_RUN in 0 1 2
do

  echo ====================== VANILLA RUN $N_RUN =============================

  python ../run.py \
  --task SpanClassificationTask \
  --dataset ImDBDataModule \
  --data_tag imdb_DNET \
  --model BertLinearCLS \
  --loss CrossEntropyLoss \
  --metrics Accuracy-AccuracyMultiClass test-CNETClassificationReport \
  --default_root_dir $LOG_ROOT/task_1/imdb/coling/vanillaCLS \
  --data_directory $DATA_ROOT/CNER_datasets/CNER_ImDB \
  --batch_size 4 \
  --gpus 1 \
  --callbacks SpanClassificationDataCollector \
  --collector_log_dir $INFERENCE_ROOT/task_1/imdb/coling/vanillaCLS  \
  --single_gpu_testing \
  --test_files unseen.json seen.json all.json  \
  --test_set_names  unseen      seen      all \
  --out_features 8 \
  --last_activation softmax \
  --gradient_clip_val 3

done