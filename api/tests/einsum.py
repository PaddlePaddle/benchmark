#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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


@benchmark_registry.register("einsum")
class EinsumConfig(APIConfig):
    def __init__(self):
        super(EinsumConfig, self).__init__("einsum")

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(EinsumConfig, self).init_from_json(filename, config_id,
                                                 unknown_dim)

        self.num_operands = 0
        for name, value in vars(self).items():
            if name.endswith("_dtype"):
                self.num_operands += 1


@benchmark_registry.register("einsum")
class PaddleEinsum(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        if config.num_operands == 2:
            x = self.variable(
                name="x", shape=config.x_shape, dtype=config.x_dtype)
            y = self.variable(
                name="y", shape=config.y_shape, dtype=config.y_dtype)
            result = paddle.einsum(config.equation, x, y)

            self.feed_list = [x, y]
            self.fetch_list = [result]
            if config.backward:
                self.append_gradients(result, [x, y])


@benchmark_registry.register("einsum")
class TorchEinsum(PytorchOpBenchmarkBase):
    def build_graph(self, config):
        if config.num_operands == 2:
            x = self.variable(
                name="x", shape=config.x_shape, dtype=config.x_dtype)
            y = self.variable(
                name="y", shape=config.y_shape, dtype=config.y_dtype)
            result = torch.einsum(config.equation, x, y).contiguous()

            self.feed_list = [x, y]
            self.fetch_list = [result]
            if config.backward:
                self.append_gradients(result, [x, y])
