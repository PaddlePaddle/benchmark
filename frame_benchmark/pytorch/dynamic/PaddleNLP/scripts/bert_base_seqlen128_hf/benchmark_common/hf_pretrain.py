#!/usr/bin/env python
# coding=utf-8
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
Fine-tuning the library models for masked language modeling (BERT...)
on a text file or a dataset without using HuggingFace Trainer.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.

import argparse
import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

import h5py
import torch
import torch.distributed
import transformers
from accelerate import Accelerator, GradScalerKwargs
from accelerate.logging import get_logger
from torch.utils.data import BatchSampler, RandomSampler, DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForPreTraining,
    SchedulerType,
    get_scheduler,
)

logger = get_logger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

torch.backends.cuda.matmul.allow_tf32 = True

class TimeCostAverage(object):
    """
    Simple tool for calcluating time average cost in the process of training and inferencing.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset the recoder state, and reset the `cnt` to zero.
        """
        self.cnt = 0
        self.total_time = 0

    def record(self, usetime):
        """
        Recoding the time cost in current step and accumulating the `cnt`.
        """
        self.cnt += 1
        self.total_time += usetime

    def get_average(self):
        """
        Returning the average time cost after the start of training.
        """
        if self.cnt == 0:
            return 0
        return self.total_time / self.cnt


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--input_dir",
        default=None,
        type=str,
        required=True,
        help="The input directory where the data will be read from.",
    )
    parser.add_argument(
        "--pure_fp16",
        action="store_true",
        help="Pure fp16, without amp.autocast decorate.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument(
        "--max_steps",
        type=int,
        default=1000000,
        help="Total number of training steps to perform.",
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
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        help="The mixed_precision.",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=0,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--dynamo_backend",
        type=str,
        default="no",
        choices=["no", "inductor"],
    )
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--max_predictions_per_seq", default=80, type=int, help="The maximum total of masked tokens in input sequence"
    )
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--scale_loss", type=float, default=2**15, help="The value of scale_loss for fp16.")

    args = parser.parse_args()

    return args

class BertPretrainingCriterion(torch.nn.Module):
    def __init__(self, vocab_size):
        super(BertPretrainingCriterion, self).__init__()
        self.vocab_size = vocab_size

    def forward(
        self, prediction_scores, seq_relationship_score, masked_lm_labels, next_sentence_labels, masked_lm_scale
    ):
        masked_lm_loss = torch.nn.functional.cross_entropy(
            prediction_scores.view(-1, self.vocab_size), masked_lm_labels.view(-1), reduction="none", ignore_index=-1
        )
        masked_lm_loss = masked_lm_loss / masked_lm_scale
        next_sentence_loss = torch.nn.functional.cross_entropy(
            seq_relationship_score.view(-1, 2), next_sentence_labels.view(-1), reduction="none"
        )
        return torch.sum(masked_lm_loss) + torch.mean(next_sentence_loss)


def set_seed(args):
    seed = args.seed + torch.distributed.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WorkerInitObj(object):
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, id):
        np.random.seed(seed=self.seed + id)
        random.seed(self.seed + id)


def create_pretraining_dataset(input_file, max_pred_length, shared_list, args, worker_init):
    train_data = PretrainingDataset(input_file=input_file, max_pred_length=max_pred_length)
    # files have been sharded, no need to dispatch again
    train_batch_sampler = BatchSampler(RandomSampler(train_data), batch_size=args.batch_size, drop_last=False)

    # DataLoader cannot be pickled because of its place.
    # If it can be pickled, use global function instead of lambda and use
    # ProcessPoolExecutor instead of ThreadPoolExecutor to prefetch.
    def _collate_data(data):
        num_fields = len(data[0])
        out = [None] * num_fields
        # input_ids, segment_ids, input_mask, masked_lm_positions,
        # masked_lm_labels, next_sentence_labels, mask_token_num
        for i in (0, 1, 2, 4, 5):
            out[i] = torch.stack([x[i] for x in data])
        _, seq_length = out[0].shape
        size = sum(len(x[3]) for x in data)
        # Padding for divisibility by 8 for fp16 or int8 usage
        if size % 8 != 0:
            size += 8 - (size % 8)
        # masked_lm_positions
        # Organize as a 1D tensor for gather or use gather_nd
        out[3] = torch.full(size=[size,], fill_value=0, dtype=torch.int32)
        
        # masked_lm_labels
        # out[4] = torch.full(size=[size, 1], fill_value=-1, dtype=torch.int64)
        mask_token_num = 0
        for i, x in enumerate(data):
            for j, pos in enumerate(x[3]):
                out[3][mask_token_num] = i * seq_length + pos
                # out[4][mask_token_num] = x[4][j]
                mask_token_num += 1
        # mask_token_num
        out.append(torch.tensor(mask_token_num, dtype=torch.float32))
        return out

    train_data_loader = DataLoader(
        dataset=train_data,
        batch_sampler=train_batch_sampler,
        collate_fn=_collate_data,
        num_workers=args.preprocessing_num_workers,
        worker_init_fn=worker_init,
    )
    return train_data_loader, input_file


class PretrainingDataset(Dataset):
    def __init__(self, input_file, max_pred_length):
        self.input_file = input_file
        self.max_pred_length = max_pred_length
        f = h5py.File(input_file, "r")
        keys = [
            "input_ids",
            "input_mask",
            "segment_ids",
            "masked_lm_positions",
            "masked_lm_ids",
            "next_sentence_labels",
        ]
        self.inputs = [np.asarray(f[key][:]) for key in keys]
        f.close()

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.inputs[0])

    def __getitem__(self, index):

        [input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_ids, next_sentence_labels] = [
            torch.from_numpy(input[index].astype(np.int64)) if indice < 5 else torch.from_numpy(
                np.asarray(input[index].astype(np.int64))) for indice, input in enumerate(self.inputs)]

        masked_lm_labels = torch.ones(input_ids.shape, dtype=torch.long) * -1
        index = self.max_pred_length
        # store number of  masked tokens in index
        padded_mask_indices = (masked_lm_positions == 0).nonzero()
        if len(padded_mask_indices) != 0:
            index = padded_mask_indices[0].item()
        masked_lm_labels[masked_lm_positions[:index]] = masked_lm_ids[:index]
        masked_lm_positions = masked_lm_positions[:index]

        return [input_ids, segment_ids, input_mask,
                masked_lm_positions, masked_lm_labels, next_sentence_labels]
        
def do_train(args):
    assert args.gradient_accumulation_steps == 1, "gradient_accumulation_steps must be equal 1!"
    if args.pure_fp16:
        args.mixed_precision = "no"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        dynamo_backend=args.dynamo_backend,
        kwargs_handlers=[GradScalerKwargs(init_scale=args.scale_loss)],
    )
    # Handle the repository creation
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    print("mixed_precision", accelerator.state.mixed_precision)
    print("dynamo_backend", accelerator.state.dynamo_backend)

    logfilepath = os.path.join(args.output_dir, f"workerlog.{accelerator.local_process_index}")
    handler = logging.FileHandler(logfilepath, mode="a", encoding="utf-8")
    logger.logger.addHandler(handler)
    logger_kwargs = {"main_process_only": False}

    logger.info(accelerator.state, **logger_kwargs)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args)
        worker_init = WorkerInitObj(args.seed + torch.distributed.get_rank())

    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.model_name_or_path:
        model = AutoModelForPreTraining.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
        )
    else:
        logger.info("Training new model from scratch", **logger_kwargs)
        model = AutoModelForPreTraining.from_config(config)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
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
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon, lr=args.learning_rate)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_steps * args.gradient_accumulation_steps,
    )
    criterion = BertPretrainingCriterion(config.vocab_size)

    # Prepare everything with our `accelerator`.
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    # Train!
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****", **logger_kwargs)
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}", **logger_kwargs)
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}", **logger_kwargs)
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}", **logger_kwargs)
    logger.info(f"  Total optimization steps = {args.max_steps}", **logger_kwargs)
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_steps), disable=True)

    pool = ThreadPoolExecutor(1)
    global_step = 0
    model.train()
    
    if args.pure_fp16:
        # is pure fp16 or not
        model.to(dtype=torch.float16)
        
    for epoch in range(sys.maxsize):
        files = [
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if os.path.isfile(os.path.join(args.input_dir, f)) and "train" in f
        ]
        files.sort()
        num_files = len(files)
        random.Random(args.seed + epoch).shuffle(files)
        f_start_id = 0

        shared_file_list = {}

        if torch.distributed.get_world_size() > num_files:
            remainder = torch.distributed.get_world_size() % num_files
            data_file = files[
                (
                    f_start_id * torch.distributed.get_world_size()
                    + torch.distributed.get_rank()
                    + remainder * f_start_id
                )
                % num_files
            ]
        else:
            data_file = files[
                (f_start_id * torch.distributed.get_world_size() + torch.distributed.get_rank()) % num_files
            ]

        train_data_loader, _ = create_pretraining_dataset(
            data_file, args.max_predictions_per_seq, shared_file_list, args, worker_init
        )

        single_file = True if f_start_id + 1 == len(files) else False

        for f_id in range(f_start_id, len(files)):
            if not single_file and f_id == f_start_id:
                continue
            if torch.distributed.get_world_size() > num_files:
                data_file = files[
                    (f_id * torch.distributed.get_world_size() + torch.distributed.get_rank() + remainder * f_id)
                    % num_files
                ]
            else:
                data_file = files[
                    (f_id * torch.distributed.get_world_size() + torch.distributed.get_rank()) % num_files
                ]

            dataset_future = pool.submit(
                create_pretraining_dataset,
                data_file,
                args.max_predictions_per_seq,
                shared_file_list,
                args,
                worker_init,
            )
            train_cost_avg = TimeCostAverage()
            reader_cost_avg = TimeCostAverage()
            total_samples = 0
            batch_start = time.time()

            for step, batch in enumerate(train_data_loader):
                train_reader_cost = time.time() - batch_start
                reader_cost_avg.record(train_reader_cost)

                with accelerator.accumulate(model):
                    outputs = model(
                        input_ids=batch[0].to(accelerator.device),
                        token_type_ids=batch[1].to(accelerator.device),
                        attention_mask=batch[2].to(accelerator.device),
                    )
                    loss = criterion(
                        prediction_scores=outputs.prediction_logits,
                        seq_relationship_score=outputs.seq_relationship_logits,
                        masked_lm_labels=batch[4].to(accelerator.device),
                        next_sentence_labels=batch[5].to(accelerator.device),
                        masked_lm_scale=batch[6].to(accelerator.device),
                    )
                    accelerator.backward(loss)
                    if args.max_grad_norm > 0 and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    total_samples += args.batch_size
                    train_run_cost = time.time() - batch_start
                    train_cost_avg.record(train_run_cost)

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                if global_step % args.logging_steps == 0:
                    logger.info(
                        "global step: %d, epoch: %d, batch: %d, loss: %f, "
                        "avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, avg_samples: %.5f, ips: %.5f sequences/sec"
                        % (
                            global_step,
                            epoch,
                            step,
                            loss.item(),
                            reader_cost_avg.get_average(),
                            train_cost_avg.get_average(),
                            total_samples / args.logging_steps,
                            total_samples / (args.logging_steps * train_cost_avg.get_average()),
                        )
                        , **logger_kwargs
                    )
                    total_samples = 0
                    train_cost_avg.reset()
                    reader_cost_avg.reset()

                if global_step % args.save_steps == 0 or global_step >= args.max_steps:
                    if accelerator.is_main_process:
                        output_dir = os.path.join(args.output_dir, "model_%d" % global_step)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = accelerator.unwrap_model(model)
                        model_to_save.save_pretrained(output_dir)

                if global_step >= args.max_steps:
                    del train_data_loader
                    return
                batch_start = time.time()

            del train_data_loader
            train_data_loader, data_file = dataset_future.result(timeout=None)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    do_train(args)
