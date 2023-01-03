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


@benchmark_registry.register("set_value")
class PaddleSetValue(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(
            name='x',
            shape=config.x_shape,
            dtype=config.x_dtype,
            stop_gradient=True)

        if config.is_tensor_value:
            value = self.variable(
                name='value',
                shape=config.value_shape,
                dtype=config.value_dtype,
                stop_gradient=True)
            x[:, 10:500:2] = value

            self.feed_list = [x, value]
            self.fetch_list = [x]
        else:
            x[:, 0:20, ::2] = 10000

            self.feed_list = [x]
            self.fetch_list = [x]


@benchmark_registry.register("set_value")
class TorchSetValue(PytorchOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(
            name='x',
            shape=config.x_shape,
            dtype=config.x_dtype,
            stop_gradient=True)

        if config.is_tensor_value:
            value = self.variable(
                name='value',
                shape=config.value_shape,
                dtype=config.value_dtype,
                stop_gradient=True)
            x[:, 10:500:2] = value

            self.feed_list = [x, value]
            self.fetch_list = [x.contiguous()]
        else:
            x[:, 0:20, ::2] = 10000

            self.feed_list = [x]
            self.fetch_list = [x.contiguous()]
