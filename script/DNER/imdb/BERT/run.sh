#!/bin/bash

for N_RUN in 1 2
do

echo ===== RUN $N_RUN ==============

python ../run.py \
          --task SequenceTaggingTask \
          --dataset ImDBDataModule \
          --data_tag imdb_DNER \
          --model DNERBertLinear \
          --loss CrossEntropyLoss \
          --metrics Accuracy-AccuracyMultiClass test-CNERClassificationReport test-CNERClassificationReportEntity \
          --default_root_dir $LOG_ROOT/task_2/imdb/coling/vanilla \
          --data_directory $DATA_ROOT/CNER_datasets/CNER_ImDB \
          --batch_size 4 \
          --gpus 1 \
          --callbacks SequenceTaggingDataCollector \
          --collector_log_dir $INFERENCE_ROOT/task_2/imdb/coling/vanilla \
          --single_gpu_testing \
          --test_files     unseen.json seen.json  \
          --test_set_names unseen      seen     \
          --classes 0   1   2   3   4  \
          --gradient_clip_val 3 \
          --out_features 17 \
          --limit_credit 3 4 \
          --force_local_load
done