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

import abc, six
import importlib


@six.add_metaclass(abc.ABCMeta)
class BenchmarkBase(object):
    def __init__(self, framework, testing_mode):
        self.name = self.__class__.__name__
        self.feed_list = None
        self.fetch_list = None
        self._backward = False
        self._framework = framework
        self._testing_mode = testing_mode
        self._task = ""

    @property
    def backward(self):
        return self._backward

    def compute_flop_and_byte(self, config):
        """ flop is used as a metric for op's performance and it is optional.
        """
        return None, None

    @abc.abstractmethod
    def build_graph(self, config=None):
        pass

    @abc.abstractmethod
    def variable(self, name, shape, dtype, value=None, stop_gradient=False):
        pass

    @abc.abstractmethod
    def layers(self, api_name, module_name=None, **kwargs):
        pass

    @abc.abstractmethod
    def append_gradients(self, targets, inputs):
        pass

    def get_running_stats(self,
                          use_gpu,
                          config,
                          runtimes,
                          walltimes=None,
                          repeat=None):
        try:
            module_name = "torch" if self._framework == "pytorch" else self._framework
            module = importlib.import_module(module_name)
            version = module.__version__
        except Exception:
            version = "none"
            print("Failed to call %s.__version__" % (self._framework))

        stats = {
            "framework": self._framework,
            "version": version,
            "name": self.name,
            "device": "GPU" if use_gpu else "CPU",
            "backward": self._backward,
            "total": runtimes
        }

        if walltimes is not None:
            stats["wall_time"] = walltimes

        if repeat is not None:
            stats["repeat"] = repeat

        try:
            flop, byte = self.compute_flop_and_byte(config)
            if flop is not None:
                stats["flop"] = flop
            if byte is not None:
                stats["byte"] = byte
        except Exception:
            print("Failed to call compute_flops_and_byte for %s." %
                  (self._framework))

        return stats
