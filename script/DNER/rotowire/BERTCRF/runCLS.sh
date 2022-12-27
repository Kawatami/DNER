#!/bin/bash

for N_RUN in 1 2 3
do

echo ===== RUN $N_RUN ==============

python ../run.py \
          --task SequenceTaggingTask \
          --dataset RotoWireDataModule \
          --data_tag roto_DNER \
          --model DNERBERTCRFCLS \
          --loss CRFLoss \
          --metrics Accuracy-AccuracyMultiClass test-CNERClassificationReport test-CNERClassificationReportEntity \
          --default_root_dir $LOG_ROOT/task_2/rotowire/coling/BERTCRFCLS \
          --data_directory $DATA_ROOT/CNER_datasets/CNER_RotoWire \
          --batch_size 4 \
          --gpus 1 \
          --callbacks SequenceTaggingDataCollector \
          --collector_log_dir $INFERENCE_ROOT/task_2/rotowire/coling/BERTCRFCLS \
          --single_gpu_testing \
          --test_files      seen_unseen.json unseen.json seen.json  \
          --test_set_names seen_unseen       unseen      seen     \
          --classes 0   1  \
          --gradient_clip_val 3
done