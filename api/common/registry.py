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

from __future__ import print_function

from common.paddle_op_benchmark import PaddleOpBenchmarkBase
from common.pytorch_api_benchmark import PytorchAPIBenchmarkBase as PytorchOpBenchmarkBase
from common.tensorflow_api_benchmark import TensorflowAPIBenchmarkBase as TensorflowOpBenchmarkBase
from common.api_param import APIConfig


class MetaData(object):
    def __init__(self):
        self.reuse = None
        self.op_type = None
        self.config_class = None
        self.paddle_class = None
        self.pytorch_class = None
        self.tensorflow_class = None

    def update(self, other):
        if other is not None:
            if self.paddle_class is None:
                self.paddle_class = other.paddle_class
            if self.pytorch_class is None:
                self.pytorch_class = other.pytorch_class
            if self.tensorflow_class is None:
                self.tensorflow_class = other.tensorflow_class

    def to_string(self):
        def _get_classname(classobj):
            return classobj.__name__ if classobj is not None else "None"

        return "reuse={}, op_type={}, config_class={}, paddle_class={}, pytorch_class={}, tensorflow_class={}".format(
            self.reuse, self.op_type,
            _get_classname(self.config_class),
            _get_classname(self.paddle_class),
            _get_classname(self.pytorch_class),
            _get_classname(self.tensorflow_class))


class BenchmarkRegistry(object):
    def __init__(self):
        self.op_meta = {}

    def register(self, filename, reuse=None, classobj=None):
        if classobj is None:
            # used as a decorator
            def deco(func_or_class):
                self._insert(filename, reuse, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        self._insert(filename, reuse, classobj)

    def get(self, filename):
        self._update_with_reuse(filename)
        info = self.op_meta.get(filename, None)
        if info is None:
            raise KeyError("No object named <{}> in op benchmark registry.".
                           format(filename))
        return info

    def _insert(self, filename, reuse, classobj):
        #print("-- Insert: {}, {}".format(filename, classobj.__name__))
        if filename not in self.op_meta.keys():
            self.op_meta[filename] = MetaData()
        self.op_meta[filename].reuse = reuse
        if issubclass(classobj, PaddleOpBenchmarkBase):
            self.op_meta[filename].paddle_class = classobj
        elif issubclass(classobj, PytorchOpBenchmarkBase):
            self.op_meta[filename].pytorch_class = classobj
        elif issubclass(classobj, TensorflowOpBenchmarkBase):
            self.op_meta[filename].tensorflow_class = classobj
        elif issubclass(classobj, APIConfig):
            self.op_meta[filename].config_class = classobj

    def _update_with_reuse(self, filename=None):
        filenames = [filename] if filename is not None else self.op_meta.keys()
        for name in filenames:
            info = self.op_meta.get(name, None)
            if info is None:
                raise KeyError(
                    "No object named <{}> in op benchmark registry.".format(
                        name))

            if info.reuse is not None:
                reuse_info = self.op_meta.get(info.reuse, None)
                info.update(reuse_info)
            elif info.config_class is None:
                info.op_type = name
                info.config_class = APIConfig

    def __str__(self):
        self._update_with_reuse()
        op_meta_str = "{\n"
        for key, value in self.op_meta.items():
            op_meta_str = op_meta_str + "  {}: {}\n".format(key,
                                                            value.to_string())
        op_meta_str = op_meta_str + "}"
        return op_meta_str


benchmark_registry = BenchmarkRegistry()
