#!/bin/bash

LOG=${1}

OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
nohup accelerate launch run_clm_no_trainer_sparse.py \
    --model_name_or_path gpt2 \
    --model_cache_dir /ssd1/models/gpt \
    --ckpt gpt2-openwebtext-ep0-ppl19.88.bin --kd \
    --config_name /ssd1/models/gpt/models--gpt2/config_no_dropout.json \
    --dataset_name openwebtext \
    --data_cache_dir /ssd1/datasets/openwebtext \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --learning_rate 1e-4 \
    --num_train_epochs 1 \
    --checkpointing_steps best \
    --prune_times 1 \
    --prune_frequency 1 \
    --num_prune_samples 128 \
    --num_steps_per_pruning 64 \
    --sparsity_per_pruning_step 0.2 \
    --output_dir ./gpt2-sparse-0.2x >> $LOG.log 2>&1 &
