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
import logging
import sys

import transformers
from ldm import (
    DataArguments,
    LatentDiffusionModel,
    LatentDiffusionTrainer,
    ModelArguments,
    TextImagePair,
)
from transformers import HfArgumentParser, TrainingArguments

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.benchmark = model_args.benchmark

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    model = LatentDiffusionModel(model_args)
    train_dataset = TextImagePair(
        file_list=data_args.file_list,
        size=data_args.resolution,
        num_records=data_args.num_records,
        buffer_size=data_args.buffer_size,
        shuffle_every_n_samples=data_args.shuffle_every_n_samples,
        interpolation="lanczos",
        tokenizer=model.tokenizer,
    )

    trainer = LatentDiffusionTrainer(
        model=model, args=training_args, train_dataset=train_dataset, tokenizer=model.tokenizer, logger=logger
    )

    trainer.train()
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    main()
