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


@benchmark_registry.register("prelu")
class PReluConfig(APIConfig):
    def __init__(self):
        super(PReluConfig, self).__init__("prelu")
        self.feed_spec = [{"range": [-1, 1]}, {"range": [-1, 1]}]


@benchmark_registry.register("prelu")
class PaddlePRelu(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name='weight',
            shape=config.weight_shape,
            dtype=config.weight_dtype)
        result = paddle.nn.functional.prelu(x=x, weight=weight)

        self.feed_list = [x, weight]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, weight])


@benchmark_registry.register("prelu")
class TorchPRelu(PytorchOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name='weight',
            shape=config.weight_shape,
            dtype=config.weight_dtype)
        result = torch.nn.functional.prelu(input=x, weight=weight)

        self.feed_list = [x, weight]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, weight])
