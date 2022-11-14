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


@benchmark_registry.register("kthvalue")
class PaddleKthvalue(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        data = self.variable(
            name='x', shape=config.x_shape, dtype=config.x_dtype)
        value, indices = paddle.kthvalue(
            x=data, k=config.k, keepdim=config.keepdim)

        self.feed_list = [data]
        self.fetch_list = [value, indices]
        if config.backward:
            self.append_gradients(value, [data])


@benchmark_registry.register("kthvalue")
class TorchKthvalue(PytorchOpBenchmarkBase):
    def build_graph(self, config):
        data = self.variable(
            name='x', shape=config.x_shape, dtype=config.x_dtype)
        value, indices = torch.kthvalue(
            input=data, k=config.k, keepdim=config.keepdim)

        self.feed_list = [data]
        self.fetch_list = [value, indices]
        if config.backward:
            self.append_gradients(value, [data])
