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


@six.add_metaclass(abc.ABCMeta)
class PaddleDynamicAPIBenchmarkBase(object):
    def __init__(self):
        self.name = self.__class__.__name__
        self.fetch_list = None
        self.run_gpu = False
        self.__backward = False
        try:
            import torch
        except Exception as e:
            sys.stderr.write(
                "Cannot import pytorch, maybe pytorch is not installed.\n")

    def build_graph(self, config=None):
        pass

    def run_graph(self, config=None):
        pass

    def variable(self, name, shape, dtype, value=None):
        assert shape is not None

        feed_value = feeder.generate_random_data(
            shape, dtype, range=None, value=value)
        paddle.disable_static()
        var = paddle.to_tensor(feed_value, stop_gradient=False)
        #print(var.stop_gradient)
        return var

    @property
    def backward(self):
        if hasattr(self, "_PaddleDynamicAPIBenchmarkBase__backward"):
            return self.__backward
        else:
            return False

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

        paddle_module_names = ["paddle", "paddle.nn.functional"]
        if module_name is not None and module_name not in paddle_module_names:
            paddle_module_names.append(module_name)

        for paddle_module_name in paddle_module_names:
            func = _import_func(paddle_module_name, api_name)
            if func is not None:
                break

        assert func is not None, "Need to specify module_name to import %s." % api_name
        result = func(**kwargs)
        return result

    def append_gradients(self, targets, inputs):
        self.__backward = True
        loss = paddle.sum(targets)
        loss.backward()
        for var in inputs:
            self.fetch_list.append(var.grad)


def run_impl(paddle_obj,
             use_gpu,
             config,
             repeat=1,
             check_output=False,
             profiler="none",
             feeder_adapter=None):

    runtimes = []
    fetches = []
    outputs = []
    if feeder_adapter is not None:
        paddle.disable_static()
        paddle_obj.feed_list = []
        for i in range(len(feeder_adapter)):
            var = paddle.to_tensor(feeder_adapter[i], stop_gradient=False)
            paddle_obj.feed_list.append(var)
    else:
        paddle_obj.build_graph(config=config)
    for i in range(repeat):
        if use_gpu:
            begin = time.time()
            paddle_obj.run_graph(config=config)
            runtimes.append(time.time() - begin)
        else:
            begin = time.time()
            paddle_obj.run_graph(config=config)
            runtimes.append(time.time() - begin)
    for var in paddle_obj.fetch_list:
        if isinstance(var, np.ndarray):
            outputs.append(var)
        else:
            outputs.append(var.numpy())

    stats = {
        "framework": "paddle",
        "version": paddle.__version__,
        "name": paddle_obj.name,
        "device": "GPU" if use_gpu else "CPU",
        "backward": paddle_obj.backward,
        "total": runtimes
    }
    return outputs, stats


def run(paddle_obj, config, args, feeder_adapter):
    paddle_obj.name = config.api_name

    outputs, stats = run_impl(
        paddle_obj=paddle_obj,
        use_gpu=args.use_gpu,
        config=config,
        repeat=args.repeat,
        check_output=args.check_output,
        profiler=args.profiler,
        feeder_adapter=feeder_adapter)
    return outputs, stats
