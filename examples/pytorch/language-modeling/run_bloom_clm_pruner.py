#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import glob
import shutil
import random

from pathlib import Path
from copy import deepcopy
from itertools import chain
from datetime import timedelta

import datasets

import torch
import torch.nn.functional as F

from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils.versions import require_version
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry

from transformers.models.bloom import BloomConfig, BloomTokenizerFast, BloomForCausalLM

# Pruning modules
import sys
ROOT = os.path.join(os.environ['HOME'], 'sparseopt')
sys.path.append(ROOT)

from sparseopt.pruner.bbs import Prune


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0",
                "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune/Prune a BLOOM model on causal language modeling task.")

    # Data
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--data_cache_dir",
        type=str,
        default=None,
        help="Directory of the dataset to be cached.",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=128,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )

    # Model
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default=None,
        help="Directory of model weights to be cached."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Path to model checkpoint."
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )

    # Train
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float,
                        default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts",
                 "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--precision_type",
        type=str,
        default=None,
        help="Whether or not to use mixed precision training. Choose from 'no','fp16','bf16 or 'fp8'."
    )

    # Global
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42,
                        help="A seed for reproducible training.")

    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
        help="multi gpu",
    )
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str,
                        help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Evaluation only.",
    )

    # Sparse
    parser.add_argument(
        "--do_pruning",
        action="store_true",
        help="Not finetune but do pruning.",
    )
    parser.add_argument(
        "--kd",
        action="store_true",
        help="Whether to perform knowledge distillation."
    )
    parser.add_argument(
        "--eval_dense",
        action="store_true",
        help="Whether to eval for dense before pruning."
    )
    parser.add_argument(
        "--target_sparsity",
        type=float,
        help="Target sparsity."
    )
    parser.add_argument(
        "--num_prune_steps",
        type=int,
        help="Total pruning steps."
    )
    parser.add_argument(
        "--prune_frequency",
        type=int,
        default=None,
        help="Pruning interval, counted by number of steps."
    )

    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError(
            "Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in [
                "csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in [
                "csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def criterion_for_kd(student_outputs, teacher_outputs):
    student_hs, student_attns, student_logits = student_outputs.hidden_states, \
        student_outputs.attentions, student_outputs.logits
    teacher_hs, teacher_attns, teacher_logits = teacher_outputs.hidden_states, \
        teacher_outputs.attentions, teacher_outputs.logits

    # Loss for logits: soft ce loss
    logit_loss = (F.softmax(teacher_logits, dim=-1) * -F.log_softmax(student_logits, dim=-1))
    # batch mean
    logit_loss = logit_loss.sum(dim=range(1, logit_loss.ndim)).mean()

    # Loss for hidden states: mse loss
    hs_loss = 0.
    for layer_stu_hs, layer_tea_hs in zip(student_hs, teacher_hs):
        hs_loss = hs_loss + F.mse_loss(layer_stu_hs, layer_tea_hs)

    # Loss for attentions: mse loss
    attn_loss = 0.
    for layer_stu_attn, layer_tea_attn in zip(student_attns, teacher_attns):
        attn_loss = attn_loss + F.mse_loss(layer_stu_attn, layer_tea_attn)
    
    return logit_loss, hs_loss, attn_loss


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_bloom_clm_pruner", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}
    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    init_kwargs = [InitProcessGroupKwargs(timeout=timedelta(seconds=180000))]
    accelerator = Accelerator(
        kwargs_handlers=init_kwargs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.precision_type, **accelerator_log_kwargs
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(
                    Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name, args.dataset_config_name,
            cache_dir=args.data_cache_dir, use_auth_token=True
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.data_cache_dir,
                split=f"train[:{args.validation_split_percentage}%]",
                use_auth_token=True
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.data_cache_dir,
                split=f"train[{args.validation_split_percentage}%:]",
                use_auth_token=True
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks

        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=args.data_cache_dir, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=args.data_cache_dir,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=args.data_cache_dir,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )
    raw_datasets['train'] = raw_datasets['train']#.select(range(1000))
    raw_datasets['validation'] = raw_datasets['validation'].select(range(1000))

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if args.config_name:
        config = BloomConfig.from_pretrained(args.config_name, cache_dir=args.model_cache_dir)
    elif args.model_name_or_path:
        config = BloomConfig.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    else:
        config = BloomConfig()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = BloomTokenizerFast.from_pretrained(args.tokenizer_name, cache_dir=args.model_cache_dir)
    elif args.model_name_or_path:
        tokenizer = BloomTokenizerFast.from_pretrained(args.model_name_or_path, cache_dir=args.model_cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    if args.model_name_or_path:
        model = BloomForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.model_cache_dir
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)
    
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    logger.info(f"\nModel structure:\n{model}\n")

    if args.ckpt:
        logger.info(f"Loading model checkpoint '{args.ckpt}'..")

        state_dict = torch.load(args.loadckpt, map_location='cpu')
        missing_keys, unexpected_keys = model.load_state_dict(state_dict)
        if len(missing_keys) and accelerator.is_local_main_process:
            logger.warning(f"Missing keys: {missing_keys}")
        if len(unexpected_keys) and accelerator.is_local_main_process:
            logger.warning(f"Unexpected keys: {unexpected_keys}")
        
        logger.info("Model checkpoint loading done!")
    
    if args.kd:
        teacher = deepcopy(model)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}

        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size

        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size]
                for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()

        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(
            f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )
    unwrapped_model = accelerator.unwrap_model(model)

    if args.kd:
        teacher = accelerator.prepare(teacher)
        teacher.eval()

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(
        args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("bloom_clm_no_trainer", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * \
        accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(
                f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            # Sorts folders by date modified, most recent checkpoint is the last
            path = dirs[-1]
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace(
                "step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            resume_step -= starting_epoch * len(train_dataloader)

    def evaluation(model, dataloader):
        model.eval()

        losses = []
        for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            # (num_devices*batch_size,)
            losses.append(accelerator.gather(loss.repeat(args.per_device_eval_batch_size)))
        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]

        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        del losses
        return perplexity

    if args.eval_dense:
        logger.info("Eval for dense..")
        ppl = evaluation(model, eval_dataloader)
        logger.info(f"Done! perplexity: {ppl}\n")

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    completed_steps = starting_epoch * num_update_steps_per_epoch

    if args.do_pruning:
        target_modules = ('query_key_value', 'self_attention.dense', 'mlp.dense_h_to_4h', 'mlp.dense_4h_to_h')

        prune_dict = {}
        for n, _ in unwrapped_model.named_parameters():
            for mod in target_modules:
                if f'{mod}.weight' in n:
                    prune_dict[n] = args.target_sparsity
        logger.info(f"Prune dict: {prune_dict}\n")

        set_up_infos = {
            'cgb': {}.fromkeys(prune_dict.keys(), 64),
            'dtype': {}.fromkeys(prune_dict.keys(), 'bf16')
        }
        pruner = Prune(
            unwrapped_model,
            group_size=64,
            deploy_device="asic",
            prune_dict=prune_dict,
            restore_sparsity=False,
            set_up_infos=set_up_infos,
            frequency=args.prune_frequency,
            sparse_step=args.num_prune_steps,
        )

        def get_sparsity():
            numel = num_zero = 0
            layer_sparse_rate = {}

            for n, p in unwrapped_model.named_parameters():
                if n in pruner._prune_dict:
                    numel += p.numel()
                    num_zero += (p == 0).sum().item()
                    layer_sparse_rate[n] = (p == 0.).float().mean().item()
            total_sparse_rate = num_zero / numel if numel else 0.

            return layer_sparse_rate, total_sparse_rate

        # Just prune 1 step, and supervise how long the model can recover to dense.
        pruner.prune()
        layer_sparsities, total_sparsity = get_sparsity()
        logger.info(f"Sparsity: {total_sparsity}")
        logger.info(f"Layer sparsities: {layer_sparsities}\n")

        logger.info("Eval after pruning..")
        ppl = evaluation(model, eval_dataloader)
        logger.info(f"Done! perplexity: {ppl}\n")

    if not args.eval_only:
        best_perplexity = float('inf')

        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()

            if args.with_tracking:
                total_loss = 0

            for step, batch in enumerate(train_dataloader):
                # We need to skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == starting_epoch and step < resume_step:
                    # if resume_step is not None and step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        completed_steps += 1

                    continue

                with accelerator.accumulate(model):
                    if args.kd:
                        batch.update(output_attentions=True, output_hidden_states=True)
                        with torch.no_grad():
                            teacher_outputs = teacher(**batch)
                    outputs = model(**batch)

                    if args.kd:
                        logit_loss, hs_loss, attn_loss = criterion_for_kd(outputs, teacher_outputs)
                        kd_loss = logit_loss + hs_loss + attn_loss
                        # TODO: Verity this
                        loss = outputs.loss + 1e-3 * kd_loss
                    else:
                        loss = outputs.loss
                    # We keep track of the loss at each epoch
                    if args.with_tracking:
                        total_loss += loss.detach().float()
                    accelerator.backward(loss)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                    if args.do_pruning:
                        pruner.prune()

                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        logger.info(f"Eval..")
                        ppl = evaluation(model, eval_dataloader)
                        model.train()
                        logger.info(f"Done! epoch {epoch}\tstep {step + 1}\tperplexity {ppl}\n")

                        output_dir = f"step_{completed_steps}-ppl_{ppl:.3f}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)

                if step % (100 * args.gradient_accumulation_steps) == 0 or step == len(train_dataloader) - 1:
                    logger.info(
                        f"\nEpoch[{epoch + 1}/{args.num_train_epochs}]\t"
                        f"Total mean loss {total_loss / (step + 1)}\t"
                        f"Loss {loss.item()}\tLr {optimizer.param_groups[0]['lr']}\t"
                    )
                    if args.kd:
                        logger.info(
                            f"Logit loss {logit_loss}\t"
                            f"Hidden state loss {hs_loss}\t"
                            f"Attention loss {attn_loss}"
                        )
                    logger.info("\n")
                
                if args.do_pruning and (step + 1) % (100 * args.gradient_accumulation_steps) == 0:
                    layer_sparsities, total_sparsity = get_sparsity()
                    logger.info(f"Sparsity: {total_sparsity}")
                    logger.info(f"Layer sparsities: {layer_sparsities}\n")
            
                    logger.info(f"Eval for sparse..")
                    ppl = evaluation(model, eval_dataloader)
                    logger.info(f"Done! epoch {epoch}\tstep {step + 1}\tperplexity {ppl}\n")

                    model.train()
                
                if completed_steps >= args.max_train_steps:
                    break

            logger.info("Eval after a training epoch..")
            perplexity = evaluation(model, eval_dataloader)
            logger.info(f"Done! epoch {epoch}\tperplexity {perplexity}\n")
            
            # Save states when better performance acheived.  
            if args.checkpointing_steps == "best" and perplexity < best_perplexity:
                best_perplexity = perplexity
                
                pattn = "epoch*-ppl*"
                output_dir = f"epoch_{epoch}-ppl_{perplexity:.3f}"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                    pattn = os.path.join(args.output_dir, pattn)
                
                # Delete previous best
                if accelerator.is_main_process:
                    for prev_best in glob.glob(pattn):
                        shutil.rmtree(prev_best)
                        
                accelerator.wait_for_everyone()
                accelerator.save_state(output_dir)
    else:
        perplexity = evaluation(model, eval_dataloader)
        logger.info(f"Perplexity: {perplexity}\n")

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()

        unwrapped_model.save_pretrained(
            args.output_dir,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save
        )

        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)

            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity}, f)


if __name__ == "__main__":
    main()
