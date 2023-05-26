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


import time

from torch.utils.data import DataLoader
from transformers.trainer import (
    PrinterCallback,
    ProgressCallback,
    Trainer,
    TrainerCallback,
    speed_metrics,
)

from .text_image_pair_dataset import TextImagePair, worker_init_fn


class AverageStatistical(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_cnt = 0
        self.time = 0

    def record(self, val, cnt=1):
        self.time += val
        self.total_cnt += cnt

    def get_average(self):
        if self.total_cnt == 0:
            return 0

        return self.time / self.total_cnt

    def get_average_per_sec(self):
        if self.time == 0.0:
            return 0.0

        return float(self.total_cnt) / self.time

    def get_total_cnt(self):
        return self.total_cnt

    def get_total_time(self):
        return self.time


class BenchmarkCallback(TrainerCallback):
    def __init__(self, benchmark=True, logger=None, **kwargs):
        self.benchmark = benchmark
        self.logger_fn = print if logger is None else logger.info

    def on_train_begin(self, args, state, control, **kwargs):
        assert args.gradient_accumulation_steps == 1 and not args.do_eval and not args.do_predict
        if self.benchmark:
            self.reader_cost_avg = AverageStatistical()

    def on_epoch_begin(self, args, state, control, **kwargs):
        if self.benchmark:
            self.epoch_start = time.time()
            self.batch_start = time.time()

    def on_step_begin(self, args, state, control, **kwargs):
        if self.benchmark:
            self.reader_cost_avg.record(time.time() - self.batch_start)

    def on_step_end(self, args, state, control, **kwargs):
        if self.benchmark:
            self.batch_start = time.time()
            if control.should_log:
                self.maybe_log_save_evaluate_start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.benchmark:
            if logs is not None and "interval_steps_per_second" in logs:
                self.batch_start = self.batch_start + (time.time() - self.maybe_log_save_evaluate_start)
                ips = logs["interval_steps_per_second"] * args.train_batch_size
                avg_batch_cost = 1 / logs["interval_steps_per_second"]
                self.logger_fn(
                    "global step %d / %d, loss: %f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, avg_samples: %.5f, ips: %.5f sample/sec"
                    % (
                        state.global_step,
                        state.max_steps,
                        logs["loss"],
                        self.reader_cost_avg.get_average(),
                        avg_batch_cost,
                        args.train_batch_size,
                        ips,
                    )
                )
                self.reader_cost_avg.reset()

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.benchmark:
            train_epoch_cost = time.time() - self.epoch_start
            self.logger_fn("train epoch: %d, epoch_cost: %.5f s" % (state.epoch, train_epoch_cost))


class LatentDiffusionTrainer(Trainer):
    def __init__(self, **kwargs):
        logger = kwargs.pop("logger", None)
        super().__init__(**kwargs)
        if self.args.benchmark is not None:
            self.add_callback(BenchmarkCallback(benchmark=self.args.benchmark, logger=logger))
            if self.args.benchmark:
                if self.args.disable_tqdm:
                    self.pop_callback(PrinterCallback)
                else:
                    self.pop_callback(ProgressCallback)

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = model(**inputs)
        return loss

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if isinstance(self.train_dataset, TextImagePair):
            return DataLoader(
                self.train_dataset,
                batch_size=self._train_batch_size,
                num_workers=self.args.dataloader_num_workers,
                worker_init_fn=None if self.args.world_size <= 1 else worker_init_fn,
            )
        else:
            return super().get_train_dataloader()

    def _inner_training_loop(self, *args, **kwargs):
        self._globalstep_last_start_time = time.time()
        return super()._inner_training_loop(*args, **kwargs)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()
            logs["global_step"] = int(self.state.global_step)

            total_train_batch_size = (
                self.args.train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
            )
            num_steps = self.state.global_step - self._globalstep_last_logged
            self.store_flos()
            logs.update(
                speed_metrics(
                    "interval",
                    self._globalstep_last_start_time,
                    num_samples=total_train_batch_size * num_steps,
                    num_steps=num_steps,
                )
            )
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self._globalstep_last_start_time = time.time()

            self.log(logs)
