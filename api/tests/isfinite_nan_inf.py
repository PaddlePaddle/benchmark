#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from common_import import *


@benchmark_registry.register("isfinite_nan_inf")
class IsfiniteNanInfConfig(APIConfig):
    def __init__(self):
        super(IsfiniteNanInfConfig, self).__init__("isfinite_nan_inf")
        self.api_name = 'isfinite'
        self.api_list = {
            'isfinite': 'isfinite',
            'isnan': 'isnan',
            'isinf': 'isinf'
        }
        # TODO(Xreki): the tf's api are different.

    #        self.api_list = {
    #            'isfinite': 'is_finite',
    #            'isnan': 'is_nan',
    #            'isinf': 'is_inf'
    #        }


@benchmark_registry.register("isfinite_nan_inf")
class PaddleIsfiniteNanInf(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = self.layers(config.api_name, x=x)

        self.feed_list = [x]
        self.fetch_list = [result]


@benchmark_registry.register("isfinite_nan_inf")
class TorchIsfiniteNanInf(PytorchOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = self.layers(config.api_name, x=x)

        self.feed_list = [x]
        self.fetch_list = [result]


@benchmark_registry.register("isfinite_nan_inf")
class TFIsfiniteNanInf(TensorflowOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        out = self.layers(config.api_name, x=x)

        self.feed_list = [x]
        self.fetch_list = [out]
