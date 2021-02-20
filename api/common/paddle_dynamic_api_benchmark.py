#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import sys
import json
import time
import abc, six
import importlib
import numpy as np
from common import special_op_list

if six.PY3:
    from . import utils
    from . import api_param
    from . import feeder
else:
    import utils
    import api_param
    import feeder

try:
    import paddle
except Exception as e:
    sys.stderr.write(
        "Cannot import paddle.fluid, maybe paddle is not installed.\n")

BEFORE_RUN = 0
IN_RUN = 1
AFTER_RUN = 2


@six.add_metaclass(abc.ABCMeta)
class PaddleDynamicAPIBenchmarkBase(object):
    def __init__(self):
        self.name = self.__class__.__name__
        self._reset()

    @abc.abstractmethod
    def build_graph(self, config=None):
        pass

    def variable(self, name, shape, dtype, value=None, stop_gradient=False):
        if self._status == BEFORE_RUN:
            if self._feed_values is not None and value is None:
                i = len(self._feed_dict)
                feed_value = self._feed_values[i]
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

    @property
    def backward(self):
        return self._backward

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
                 check_output=False,
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
        fetches = []

        self._status = IN_RUN
        for i in range(repeat):
            begin = time.time()
            outputs = _run_main_iter()
            runtimes.append(time.time() - begin)

        self._status = AFTER_RUN
        stats = {
            "framework": "paddle",
            "version": paddle.__version__,
            "name": self.name,
            "device": "GPU" if use_gpu else "CPU",
            "backward": self._backward,
            "total": runtimes
        }
        return outputs, stats

    def run(self, config, args, feeder_adapter=None):
        paddle.disable_static()
        self.name = config.api_name

        self._reset()
        self._feed_spec = feeder.copy_feed_spec(config.feed_spec)
        self._need_fetch = args.task == "accuracy"
        if feeder_adapter:
            self._feed_values = feeder_adapter.to_paddle()
        outputs, stats = self.run_impl(
            use_gpu=args.use_gpu,
            config=config,
            repeat=args.repeat,
            check_output=args.check_output,
            profiler=args.profiler,
            feeder_adapter=feeder_adapter)
        return outputs, stats

    def _reset(self):
        self.feed_list = None
        self.fetch_list = None
        self._feed_spec = None
        self._feed_values = None
        self._feed_list = []
        self._backward = False
        self._status = BEFORE_RUN
        self._feed_dict = {}
        self._layers_function = None
        self._ones_like_targets = []
