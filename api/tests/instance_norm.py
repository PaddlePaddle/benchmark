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


@benchmark_registry.register("instance_norm")
class InstanceNormConfig(APIConfig):
    def __init__(self):
        super(InstanceNormConfig, self).__init__('instance_norm')
        self.run_tf = False


@benchmark_registry.register("instance_norm")
class PaddleInstanceNorm(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(
            name='input', shape=config.x_shape, dtype=config.x_dtype)
        result = paddle.nn.functional.instance_norm(x=x, eps=config.eps)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


@benchmark_registry.register("instance_norm")
class TorchInstanceNorm(PytorchOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(
            name='input', shape=config.x_shape, dtype=config.x_dtype)
        result = torch.nn.functional.instance_norm(input=x, eps=config.eps)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])
