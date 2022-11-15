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


@benchmark_registry.register("isclose")
class IscloseConfig(APIConfig):
    def __init__(self):
        super(IscloseConfig, self).__init__('isclose')
        self.feed_spec = [{"range": [-1, 1]}, {"range": [-1, 1]}]


@benchmark_registry.register("isclose")
class PaddleIsclose(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)
        result = paddle.isclose(
            x=x,
            y=y,
            rtol=config.rtol,
            atol=config.atol,
            equal_nan=config.equal_nan)

        self.feed_list = [x]
        self.fetch_list = [result]


@benchmark_registry.register("isclose")
class TorchIsclose(PytorchOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)
        result = torch.isclose(
            input=x,
            other=y,
            rtol=config.rtol,
            atol=config.atol,
            equal_nan=config.equal_nan)

        self.feed_list = [x]
        self.fetch_list = [result]
