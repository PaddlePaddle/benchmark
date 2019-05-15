#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import os
import random

import math

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid.executor import Executor
import paddle.fluid.profiler as profiler

import reader

import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
sys.path.append('..')
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from args import *
import lm_model
import logging
import pickle

SEED = 123


def get_current_model_para(train_prog, train_exe):
    param_list = train_prog.block(0).all_parameters()
    param_name_list = [p.name for p in param_list]

    vals = {}
    for p_name in param_name_list:
        p_array = np.array(fluid.global_scope().find_var(p_name).get_tensor())
        vals[p_name] = p_array

    return vals


def save_para_npz(train_prog, train_exe):
    print("begin to save model to model_base")
    param_list = train_prog.block(0).all_parameters()
    param_name_list = [p.name for p in param_list]

    vals = {}
    for p_name in param_name_list:
        p_array = np.array(fluid.global_scope().find_var(p_name).get_tensor())
        vals[p_name] = p_array

    emb = vals["embedding_para"]
    print("begin to save model to model_base")
    np.savez("mode_base", **vals)


def main():
    args = parse_args()
    model_type = args.model_type
    rnn_model = args.rnn_model

    logger = logging.getLogger("lm")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    vocab_size = 10000
    if model_type == "test":
        num_layers = 1
        batch_size = 2
        hidden_size = 10
        num_steps = 3
        init_scale = 0.1
        max_grad_norm = 5.0
        epoch_start_decay = 1
        max_epoch = 1
        dropout = 0.0
        lr_decay = 0.5
        base_learning_rate = 1.0
    elif model_type == "small":
        num_layers = 2
        batch_size = 20
        hidden_size = 200
        num_steps = 20
        init_scale = 0.1
        max_grad_norm = 5.0
        epoch_start_decay = 4
        max_epoch = 13
        dropout = 0.0
        lr_decay = 0.5
        base_learning_rate = 1.0
    elif model_type == "medium":
        num_layers = 2
        batch_size = 20
        hidden_size = 650
        num_steps = 35
        init_scale = 0.05
        max_grad_norm = 5.0
        epoch_start_decay = 6
        max_epoch = 39
        dropout = 0.5
        lr_decay = 0.8
        base_learning_rate = 1.0
    elif model_type == "large":
        num_layers = 2
        batch_size = 20
        hidden_size = 1500
        num_steps = 35
        init_scale = 0.04
        max_grad_norm = 10.0
        epoch_start_decay = 14
        max_epoch = 55
        dropout = 0.65
        lr_decay = 1.0 / 1.15
        base_learning_rate = 1.0
    else:
        print("model type not support")
        return

    if not args.save_model_dir:
        save_model_dir = model_type + "_models"
        if args.use_gpu:
            save_model_dir = "gpu_" + save_model_dir
        else:
            save_model_dir = "cpu_" + save_model_dir
        if args.inference_only:
            save_model_dir = "infer_" + save_model_dir
        else:
            save_model_dir = "train_" + save_model_dir
    else:
        save_model_dir = args.save_model_dir

    if args.batch_size > 0:
        batch_size = args.batch_size

    if args.max_epoch > 0:
        max_epoch = args.max_epoch

    if args.profile:
        print(
            "\nProfiler is enabled, only 1 epoch will be ran (set max_epoch = 1).\n"
        )
        max_epoch = 1

    main_program = fluid.Program()
    startup_program = fluid.Program()
    if args.enable_ce:
        startup_program.random_seed = SEED

    with fluid.program_guard(main_program, startup_program):
        # Training process
        loss, last_hidden, last_cell, feed_order = lm_model.lm_model(
            hidden_size,
            vocab_size,
            batch_size,
            num_layers=num_layers,
            num_steps=num_steps,
            init_scale=init_scale,
            dropout=dropout, 
            rnn_model=rnn_model)

        # clone from default main program and use it as the validation program
        inference_program = fluid.default_main_program().clone(for_test=True)

        #print(inference_program)

        fluid.clip.set_gradient_clip(
            clip=fluid.clip.GradientClipByGlobalNorm(clip_norm=max_grad_norm))

        learning_rate = fluid.layers.create_global_var(
            name="learning_rate",
            shape=[1],
            value=1.0,
            dtype='float32',
            persistable=True)

        optimizer = fluid.optimizer.SGD(learning_rate=learning_rate)
        optimizer.minimize(loss)

    place = core.CUDAPlace(0) if args.use_gpu else core.CPUPlace()
    exe = Executor(place)
    exe.run(startup_program)

    device_count = fluid.core.get_cuda_device_count()

    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = device_count
    exec_strategy.use_experimental_executor = False
    exec_strategy.num_iteration_per_drop_scope = 100

    build_strategy = fluid.BuildStrategy()
    build_strategy.enable_inplace = True
    build_strategy.memory_optimize = False

    build_strategy.remove_unnecessary_lock = True
    build_strategy.enable_sequential_execution = False
    build_strategy.cache_runtime_context = True
    build_strategy.cache_expected_kernel = True
    build_strategy.fuse_all_optimizer_ops = True

    if args.parallel:
        train_program = fluid.compiler.CompiledProgram(main_program).with_data_parallel(
            loss_name=loss.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)
    else:
        train_program = fluid.compiler.CompiledProgram(main_program)

    data_path = args.data_path
    print("begin to load data")
    raw_data = reader.ptb_raw_data(data_path)
    print("finished load data")
    train_data, valid_data, test_data, _ = raw_data

    def prepare_input(batch,
                      init_hidden=None,
                      init_cell=None,
                      epoch_id=0,
                      with_lr=True,
                      device_count=1):
        x, y = batch
        new_lr = base_learning_rate * (lr_decay**max(
            epoch_id + 1 - epoch_start_decay, 0.0))
        res = {}
        if device_count > 1 and args.parallel:
            lr = np.ones((device_count), dtype='float32') * new_lr
            x = x.reshape((-1, num_steps, 1))
            y = y.reshape((-1, 1))
        else:
            lr = np.ones((1), dtype='float32') * new_lr
            x = x.reshape((-1, num_steps, 1))
            y = y.reshape((-1, 1))

        res['x'] = x
        res['y'] = y
        if init_hidden is not None:
            res['init_hidden'] = init_hidden
        if init_cell is not None:
            res['init_cell'] = init_cell
        if with_lr:
            res['learning_rate'] = lr

        return res

    def eval(data):
        if args.inference_only and args.init_params_path:
            dirname = args.init_params_path
            filename = None
            if not os.path.isdir(args.init_params_path):
                dirname = os.path.dirname(args.init_params_path)
                filename = os.path.basename(args.init_params_path)
            fluid.io.load_persistables(
                exe, dirname, main_program=main_program, filename=filename)
            print("Load parameters from: %s." % args.init_params_path)

        batch_times = []
        start_time = time.time()
        # when eval the batch_size set to 1
        eval_data_iter = reader.get_data_iter(data, batch_size, num_steps)
        total_loss = 0.0
        iters = 0
        init_hidden = np.zeros(
            (num_layers, batch_size, hidden_size), dtype='float32')
        init_cell = np.zeros(
            (num_layers, batch_size, hidden_size), dtype='float32')
        for batch_id, batch in enumerate(eval_data_iter):
            input_data_feed = prepare_input(
                batch, init_hidden, init_cell, epoch_id=0, with_lr=False)

            batch_start_time = time.time()
            # eval should not run the grad op and change the parameters.
            # use Executor to eval
            fetch_outs = exe.run(
                program=inference_program,
                feed=input_data_feed,
                fetch_list=[loss.name, last_hidden.name, last_cell.name],
                use_program_cache=True)
            batch_times.append(time.time() - batch_start_time)

            cost_train = np.array(fetch_outs[0])
            init_hidden = np.array(fetch_outs[1])
            init_cell = np.array(fetch_outs[2])

            total_loss += cost_train
            iters += num_steps

        ppl = np.exp(total_loss / iters)

        eval_time_total = time.time() - start_time
        eval_time_run = np.sum(batch_times)

        # Benchmark
        if args.inference_only:
            print("\n======== Benchmark Result ========")
            print(
                "Eval batch_size: %d; Time (total): %.5f s; Time (only run): %.5f s; ppl: %.5f"
                % (batch_size, eval_time_total, eval_time_run, ppl[0]))
            print("")

            # Save the inference model for C++ inference purpose
            fluid.io.save_inference_model(
                save_model_dir,
                feed_order, [loss, last_hidden, last_cell],
                exe,
                main_program=inference_program,
                model_filename="model",
                params_filename="params")
            print("Save inference model to: %s." % save_model_dir)

        return ppl

    def train_an_epoch(epoch_id, batch_times):
        # get train epoch size
        num_batchs = len(train_data) // batch_size
        epoch_size = (num_batchs - 1) // num_steps
        if args.profile:
            log_interval = 1
        else:
            log_interval = max(1, epoch_size // 10)

        data_iter_size = batch_size
        if device_count > 1 and args.parallel:
            data_iter_size = batch_size * device_count
        train_data_iter = reader.get_data_iter(train_data, data_iter_size,
                                               num_steps)

        total_loss = 0
        iters = 0
        if device_count > 1 and args.parallel:
            init_hidden = np.zeros(
                (num_layers * device_count, batch_size, hidden_size),
                dtype='float32')
            init_cell = np.zeros(
                (num_layers * device_count, batch_size, hidden_size),
                dtype='float32')
        else:
            init_hidden = np.zeros(
                (num_layers, batch_size, hidden_size), dtype='float32')
            init_cell = np.zeros(
                (num_layers, batch_size, hidden_size), dtype='float32')
        for batch_id, batch in enumerate(train_data_iter):
            if batch_id == 0:
                input_data_feed = prepare_input(
                    batch,
                    init_hidden=init_hidden,
                    init_cell=init_cell,
                    epoch_id=epoch_id,
                    device_count=device_count)
            else:
                input_data_feed = prepare_input(
                    batch,
                    init_hidden=None,
                    init_cell=None,
                    epoch_id=epoch_id,
                    device_count=device_count)

            batch_start_time = time.time()
            fetch_outs = exe.run(train_program,
                                 feed=input_data_feed,
                                 fetch_list=[loss.name, "learning_rate"],
                use_program_cache=True)
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)

            cost_train = np.array(fetch_outs[0])
            lr = np.array(fetch_outs[1])

            total_loss += cost_train
            iters += num_steps
            if batch_id > 0 and batch_id % log_interval == 0:
                ppl = np.exp(total_loss / iters)
                print("-- Epoch:[%d]; Batch:[%d]; Time: %.5f s; ppl: %.5f, lr: %.5f" % (epoch_id, batch_id, batch_time, ppl[0], lr[0]))

            if args.profile:
                if batch_id == 1:
                    profiler.reset_profiler()
                elif batch_id >= 11:
                    break

        ppl = np.exp(total_loss / iters)
        return ppl


    def train():
        total_time = 0.0
        for epoch_id in range(max_epoch):
            batch_times = []
            epoch_start_time = time.time()
            train_ppl = train_an_epoch(epoch_id, batch_times)
            epoch_time = time.time() - epoch_start_time
            total_time += epoch_time
            print("\nTrain epoch:[%d]; epoch Time: %.5f; ppl: %.5f; avg_time: %.5f steps/s \n" %
                  (epoch_id, epoch_time, train_ppl[0], len(batch_times) / sum(batch_times)))

            # FIXME(zjl): ppl[0] increases as batch_size increases. 
            # We should find a better way to calculate ppl by normalizing batch_size. 
            if device_count == 1 and batch_size <= 20 and epoch_id == 0 and train_ppl[
                    0] > 1000:
                # for bad init, after first epoch, the loss is over 1000
                # no more need to continue
                print(
                    "Parameters are randomly initialized and not good this time because the loss is over 1000 after the first epoch."
                )
                print("Abort this training process and please start again.")
                return

            if epoch_id == max_epoch - 1 and args.enable_ce:
                # kpis
                print("ptblm\tlstm_language_model_duration\t%s" %
                      (total_time / max_epoch))
                print("ptblm\tlstm_language_model_loss\t%s" % train_ppl[0])

            if not args.profile:
                # NOTE(zjl): sometimes we have not enough data for eval if batch_size is large, i.e., 2100
                # Just skip to avoid error
                def is_valid_data(data, batch_size, num_steps):
                    data_len = len(data)
                    batch_len = data_len // batch_size
                    epoch_size = (batch_len - 1) // num_steps
                    return epoch_size >= 1

                valid_data_valid = is_valid_data(valid_data, batch_size,
                                                 num_steps)

                test_data_valid = is_valid_data(test_data, batch_size,
                                                num_steps)

                if valid_data_valid and test_data_valid:
                    valid_ppl = eval(valid_data)
                    print("Valid ppl: %.5f" % valid_ppl[0])

                    test_ppl = eval(test_data)
                    print("Test ppl: %.5f" % test_ppl[0])
                else:
                    if not valid_data_valid:
                        print(
                            'WARNING: length of valid_data is {}, which is not enough for batch_size {} and num_steps {}'.
                            format(len(valid_data), batch_size, num_steps))

                    if not test_data_valid:
                        print(
                            'WARNING: length of test_data is {}, which is not enough for batch_size {} and num_steps {}'.
                            format(len(test_data), batch_size, num_steps))

                filename = "params_%05d" % epoch_id
                fluid.io.save_persistables(
                    executor=exe,
                    dirname=save_model_dir,
                    main_program=main_program,
                    filename=filename)
                print("Saved model to: %s/%s.\n" % (save_model_dir, filename))

    if args.profile:
        if args.use_gpu:
            profiler.start_profiler("All")
            if not args.inference_only:
                profile_filename = "train_padding_rnn.gpu.profile"
                train()
            else:
                profile_filename = "infer_padding_rnn.gpu.profile"
                eval(test_data)
            profiler.stop_profiler("total", profile_filename)
        else:
            profiler.start_profiler("CPU")
            if not args.inference_only:
                profile_filename = "train_padding_rnn.cpu.profile"
                train()
            else:
                profile_filename = "infer_padding_rnn.cpu.profile"
                eval(test_data)
            profiler.stop_profiler("total", profile_filename)
    else:
        if not args.inference_only:
            train()
        else:
            eval(test_data)


if __name__ == '__main__':
    main()
