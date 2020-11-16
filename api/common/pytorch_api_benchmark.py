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
    import torch
except Exception as e:
    sys.stderr.write(
        "Cannot import pytorch, maybe pytorch is not installed.\n")


@six.add_metaclass(abc.ABCMeta)
class PytorchAPIBenchmarkBase(object):
    def __init__(self):
        self.name = self.__class__.__name__
        self.fetch_list = None
        self.run_gpu = True
        self._feed_list = []
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

        #if value is None:
        #    i = len(self._feed_list)
        #    range = self._feed_list[i].get("range", None)
        #else:
        #    range = None

        feed_value = feeder.generate_random_data(
            shape, dtype, range=None, value=value)
        var = torch.tensor(feed_value, requires_grad=True)
        #var.requires_grad_(True)
        #print(var)

        if self.run_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            var = var.to(device)
        var.retain_grad()

        if value is None:
            # When value is None, the variable is need to feed data.
            self._feed_list.append(feed_value)
        return var

    @property
    def backward(self):
        if hasattr(self, "_PytorchAPIBenchmarkBase__backward"):
            return self.__backward
        else:
            return False

    def layers(self, api_name, module_name=None, **kwargs):
        def _import_func(torch_module_name, api_name):
            try:
                module = importlib.import_module(torch_module_name)
                func = getattr(module, api_name)
                print("Successly import %s.%s" % (torch_module_name, api_name))
                return func
            except Exception:
                print("Failed to import %s.%s" % (torch_module_name, api_name))
            return None

        torch_module_names = ["torch"]
        if module_name is not None and module_name not in torch_module_names:
            torch_module_names.append(module_name)

        for torch_module_name in torch_module_names:
            func = _import_func(torch_module_name, api_name)
            if func is not None:
                break

        assert func is not None, "Need to specify module_name to import %s." % api_name
        result = func(**kwargs)
        return result

    def get_feeder(self):
        return self._feed_list
        #return self.feed_list

    def append_gradients(self, targets, inputs):
        self.__backward = True
        #print(self.__backward)
        #print(targets)
        #loss = torch.ones(targets.shape)
        loss = targets.sum()
        loss.backward()
        loss.retain_grad()
        for var in self.feed_list:
            #print("Var: ", var)
            #print("Gradients: ", loss.grad)
            #print("Gradients: ", var.grad)
            self.fetch_list.append(var.grad)
        #print(loss.grad)
        #print("Gradients: ", loss.grad)
        #if isinstance(loss.grad, list):
        #    for grad in loss.grad:
        #        self.fetch_vars.append(grad)
        #else:
        #    self.fetch_vars.append(loss.grad)


def run_impl(torch_obj,
             use_gpu,
             config,
             repeat=1,
             check_output=False,
             profiler="none"):

    runtimes = []
    fetches = []
    outputs = []
    torch_obj.build_graph(config=config)
    #print("repeat:")
    #print(repeat)
    for i in range(repeat):
        if use_gpu:
            begin = time.time()
            #print("iteratable test")
            torch_obj.run_graph(config=config)
            #print("iteratable test 2")
            runtimes.append(time.time() - begin)
        else:
            begin = time.time()
            torch_obj.run_graph(config=config)
            runtimes.append(time.time() - begin)
    #print("iteratable test 3")
    for var in torch_obj.fetch_list:
        outputs.append(var.to("cpu").detach().numpy())
    #print(outputs)
    #print("iteratable test 4")

    stats = {
        "framework": "pytorch",
        "version": torch.__version__,
        "name": torch_obj.name,
        "device": "GPU" if use_gpu else "CPU",
        "backward": torch_obj.backward,
        "total": runtimes
    }
    #print(stats)
    #print("iteratable test 5")
    return outputs, stats


def run(torch_obj, config, args):
    torch_obj.name = config.api_name

    #print("run before")
    outputs, stats = run_impl(
        torch_obj=torch_obj,
        use_gpu=args.use_gpu,
        config=config,
        repeat=args.repeat,
        check_output=args.check_output,
        profiler=args.profiler)
    #print("run after")
    return outputs, stats
