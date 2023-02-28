#!/bin/bash

LOG=${1}

OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
nohup accelerate launch run_clm_no_trainer.py \
    --model_name_or_path gpt2 \
    --model_cache_dir /ssd1/models/gpt \
    --dataset_name openwebtext \
    --data_cache_dir /ssd1/datasets/openwebtext \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --output_dir ./gpt2-finetuned >> $LOG.log 2>&1 &
