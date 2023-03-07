#!/bin/bash

LOG=${1}

OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
nohup accelerate launch run_clm_no_trainer_sparse.py \
    --model_name_or_path gpt2 \
    --model_cache_dir /ssd1/models/gpt \
    --config_name /ssd1/models/gpt/models--gpt2/config_no_dropout.json \
    --dataset_name openwebtext \
    --data_cache_dir /ssd1/datasets/openwebtext \
    --preprocessing_num_workers 8 \
    --max_train_samples 1280 --max_eval_samples 1280 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --num_train_epochs 1 \
    --checkpointing_steps best \
    --prune_times 3 \
    --prune_frequency 16 \
    --num_prune_samples 1 \
    --num_steps_per_pruning 16 \
    --sparsity_per_pruning_step 0.2 \
    --output_dir ./gpt2-sparse >> $LOG.log 2>&1 &
