# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import math
import types
from dataclasses import dataclass, field
from typing import List, Optional

import torch
from transformers.trainer import TrainingArguments
import os
from transformers.utils.logging import get_logger

__all__ = [
    "SDTrainingArguments",
    "SDModelArguments",
    "SDDataArguments",
]

logger = get_logger("transformers")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if not isinstance(v, str):
        v = str(v)
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Not supported value: {}".format(v))


if not str2bool(os.getenv("FLAG_SDP", "True")):
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        torch.nn.functional.scaled_dot_product_attention_ = torch.nn.functional.scaled_dot_product_attention
        del torch.nn.functional.scaled_dot_product_attention
        print(
            "Removed `torch.nn.functional.scaled_dot_product_attention`, we will use default attention implement."
        )


@dataclass
class SDTrainingArguments(TrainingArguments):
    image_logging_steps: int = field(
        default=1000, metadata={"help": "Log image every X steps."})
    recompute: bool = field(
        default=False,
        metadata={"help": "Whether or not run recompute."}, )
    benchmark: bool = field(
        default=False,
        metadata={"help": "Whether or not run benchmark."}, )
    report_to: Optional[List[str]] = field(
        default_factory=lambda: ["custom_visualdl"],
        metadata={
            "help":
            "The list of integrations to report the results and logs to."
        }, )
    resolution: int = field(
        default=512,
        metadata={
            "help":
            "The resolution for input images, all the images in the train/validation dataset will be resized to this resolution."
        }, )
    use_ema: bool = field(
        default=False, metadata={"help": "Whether or not use ema"})
    enable_xformers_memory_efficient_attention: bool = field(
        default=False,
        metadata={"help": "enable_xformers_memory_efficient_attention."})
    only_save_updated_model: bool = field(
        default=True,
        metadata={"help": "Whether or not save only_save_updated_model"})
    log_level: str = field(
        default="info",
        metadata={"help": "log_level."}, )

    def __post_init__(self):
        super().__post_init__()
        self.image_logging_steps = ((math.ceil(
            self.image_logging_steps / self.logging_steps) * self.logging_steps)
                                    if self.image_logging_steps > 0 else -1)
        self.use_ema = str2bool(os.getenv("FLAG_USE_EMA",
                                          "False")) or self.use_ema
        self.enable_xformers_memory_efficient_attention = (
            str2bool(os.getenv("FLAG_XFORMERS", "False")) or
            self.enable_xformers_memory_efficient_attention)
        self.recompute = str2bool(os.getenv("FLAG_RECOMPUTE",
                                            "False")) or self.recompute
        self.gradient_checkpointing = self.gradient_checkpointing or self.recompute
        self.benchmark = str2bool(os.getenv("FLAG_BENCHMARK",
                                            "False")) or self.benchmark

    def print_config(self, args=None, key=""):
        """
        print all config values.
        """
        logger.info("=" * 60)
        if args is None:
            args = self
            key = "Training"

        logger.info("{:^40}".format("{} Configuration Arguments".format(key)))
        logger.info("{:30}: {}".format("torch version", torch.__version__))
        logger.info("{:30}: {}".format("torch commit id",
                                       torch.version.git_version))

        for a in dir(args):
            if a[:2] != "__":  # don't print double underscore methods
                v = getattr(args, a)
                if not isinstance(v, types.MethodType):
                    logger.info("{:30}: {}".format(a, v))

        logger.info("")


@dataclass
class SDModelArguments:
    vae_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "vae_name_or_path"})
    text_encoder_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "text_encoder_name_or_path"})
    unet_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "unet_name_or_path"})
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "Pretrained tokenizer name or path if not the same as pretrained_model_name_or_path"
        }, )
    pretrained_model_name_or_path: str = field(
        default="CompVis/stable-diffusion-v1-4",
        metadata={
            "help":
            "Path to pretrained model or model, when we want to resume training."
        }, )
    model_max_length: int = field(
        default=77, metadata={"help": "Pretrained tokenizer model_max_length"})
    prediction_type: str = field(
        default="epsilon",
        metadata={
            "help":
            "prediction_type, prediction type of the scheduler function, one of `epsilon` (predicting the noise of the diffusion process), `sample` (directly predicting the noisy sample`) or `v_prediction` (see section 2.4 https://imagen.research.google/video/paper.pdf)"
        }, )
    num_inference_steps: int = field(
        default=50, metadata={"help": "num_inference_steps"})
    train_text_encoder: bool = field(
        default=False, metadata={"help": "Whether or not train text encoder"})

    noise_offset: float = field(
        default=0, metadata={"help": "The scale of noise offset."})
    snr_gamma: Optional[float] = field(
        default=None,
        metadata={
            "help":
            "SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. More details here: https://arxiv.org/abs/2303.09556."
        }, )
    input_perturbation: Optional[float] = field(
        default=0,
        metadata={"help": "The scale of input perturbation. Recommended 0.1."})


@dataclass
class SDDataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training.
    """

    file_list: str = field(
        default="./data/filelist/train.filelist.list",
        metadata={"help": "The name of the file_list."})
    num_records: int = field(default=10000000, metadata={"help": "num_records"})
    buffer_size: int = field(
        default=100,
        metadata={"help": "Buffer size"}, )
    shuffle_every_n_samples: int = field(
        default=5,
        metadata={"help": "shuffle_every_n_samples."}, )
    interpolation: str = field(
        default="lanczos",
        metadata={"help": "interpolation method"}, )
