#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import sys
import json
import time
import importlib
import numpy as np

from common import utils
from common import api_param
from common import feeder
from common import special_op_list
from common.benchmark import BenchmarkBase
from common.paddle_api_benchmark import profile_context

try:
    import paddle
except Exception as e:
    sys.stderr.write(
        "Cannot import paddle.fluid, maybe paddle is not installed.\n")


class DynamicHelper(object):
    BEFORE_RUN = 0
    IN_RUN = 1
    AFTER_RUN = 2

    def __init__(self):
        self._feed_spec = None
        self._feed_values = None
        self._feed_dict = {}
        self._ones_like_targets = []
        self._status = DynamicHelper.BEFORE_RUN

    def set_feed_spec(self, feed_spec):
        self._feed_spec = feeder.copy_feed_spec(feed_spec)

    def set_feed_values(self, feed_values):
        self._feed_values = feed_values

    def switch_status(self, status=None):
        if status is not None:
            self._status = status
        else:
            self._status = (self._status + 1) % 3

    def variable(self, name, shape, dtype, value=None, stop_gradient=False):
        if self._status == DynamicHelper.BEFORE_RUN:
            if self._feed_values is not None and value is None:
                i = len(self._feed_dict)
                feed_value = feeder.check_shape_and_dtype(
                    shape=shape, dtype=dtype, value=self._feed_values[i])
            else:
                assert shape is not None

                if self._feed_spec is not None and value is None:
                    i = len(self._feed_dict)
                    range = self._feed_spec[i].get("range", None)
                else:
                    range = None
                feed_value = feeder.generate_random_data(
                    shape, dtype, range=range, value=value)
            var = paddle.to_tensor(feed_value, stop_gradient=stop_gradient)
            self._feed_dict[name] = var
        else:
            var = self._feed_dict[name]
        return var

    def generate_gradients(self, targets, inputs):
        if not isinstance(targets, list):
            if len(self._ones_like_targets) == 0:
                ones_like_targets = paddle.ones_like(targets)
                self._ones_like_targets.append(ones_like_targets)
            else:
                ones_like_targets = self._ones_like_targets[0]
        else:
            ones_like_targets = None
        gradients = paddle.grad(
            outputs=targets, inputs=inputs, grad_outputs=ones_like_targets)
        return gradients


class PaddleOpBenchmarkBase(BenchmarkBase):
    def __init__(self, testing_mode):
        super(PaddleOpBenchmarkBase, self).__init__(testing_mode)
        self._layers_function = None
        if testing_mode == "dynamic":
            self._helper = DynamicHelper()

    def variable(self, name, shape, dtype, value=None, stop_gradient=False):
        return self._helper.variable(name, shape, dtype, value, stop_gradient)

    def layers(self, api_name, module_name=None, **kwargs):
        def _import_func(paddle_module_name, api_name):
            try:
                module = importlib.import_module(paddle_module_name)
                func = getattr(module, api_name)
                print("Successly import %s.%s" %
                      (paddle_module_name, api_name))
                return func
            except Exception:
                print("Failed to import %s.%s" %
                      (paddle_module_name, api_name))
            return None

        if self._layers_function is None:
            paddle_module_names = ["paddle", "paddle.nn.functional"]
            if module_name is not None and module_name not in paddle_module_names:
                paddle_module_names.append(module_name)

            for paddle_module_name in paddle_module_names:
                func = _import_func(paddle_module_name, api_name)
                if func is not None:
                    break

            assert func is not None, "Need to specify module_name to import %s." % api_name
            self._layers_function = func

        result = self._layers_function(**kwargs)
        return result

    def append_gradients(self, targets, inputs):
        gradients = self._helper.generate_gradients(targets, inputs)
        self._backward = True
        if isinstance(gradients, list):
            for grad in gradients:
                self.fetch_list.append(grad)
        else:
            self.fetch_list.append(gradients)

    def run_impl(self,
                 use_gpu,
                 config,
                 repeat=1,
                 profiler="none",
                 feeder_adapter=None):
        def _run_main_iter():
            self.build_graph(config=config)
            if use_gpu:
                paddle.fluid._cuda_synchronize(paddle.fluid.CUDAPlace(0))

            outputs = None
            if self._need_fetch:
                outputs = []
                for var in self.fetch_list:
                    if isinstance(var, np.ndarray):
                        outputs.append(var)
                    else:
                        outputs.append(var.numpy())
            return outputs

        # warmup run
        _run_main_iter()

        runtimes = []

        self._helper.switch_status()
        with profile_context(self.name, use_gpu, profiler):
            for i in range(repeat):
                begin = time.time()
                outputs = _run_main_iter()
                runtimes.append(time.time() - begin)

        self._helper.switch_status()
        stats = self._get_output_stats(use_gpu, config, runtimes)
        return outputs, stats

    def run(self, config, args, feeder_adapter=None):
        paddle.disable_static()
        self.name = config.api_name

        self._helper.set_feed_spec(config.feed_spec)
        self._helper.switch_status(status=DynamicHelper.BEFORE_RUN)

        self._need_fetch = args.task == "accuracy"
        if feeder_adapter:
            self._helper.set_feed_values(feeder_adapter.to_paddle())
        outputs, stats = self.run_impl(
            use_gpu=args.use_gpu,
            config=config,
            repeat=args.repeat,
            profiler=args.profiler,
            feeder_adapter=feeder_adapter)
        return outputs, stats

    def _get_output_stats(self, use_gpu, config, runtimes):
        stats = {
            "framework": "paddle",
            "version": paddle.__version__,
            "name": self.name,
            "device": "GPU" if use_gpu else "CPU",
            "backward": self._backward,
            "total": runtimes
        }

        flop, byte = self.compute_flop_and_byte(config)
        if flop is not None:
            stats["flop"] = flop
        if byte is not None:
            stats["byte"] = byte
        return stats
