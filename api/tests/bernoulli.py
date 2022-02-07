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


@benchmark_registry.register("bernoulli")
class PaddleBernoulli(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = paddle.bernoulli(x=x)

        self.feed_list = [x]
        self.fetch_list = [result]


@benchmark_registry.register("bernoulli")
class TorchBernoulli(PytorchOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = torch.bernoulli(input=x)

        self.feed_list = [x]
        self.fetch_list = [result]


@benchmark_registry.register("bernoulli")
class TFBernoulli(TensorflowOpBenchmarkBase):
    def build_graph(self, config):
        import tensorflow_probability as tfp

        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        b = tfp.distributions.Bernoulli(probs=x)
        result = b.sample()

        self.feed_list = [x]
        self.fetch_list = [result]
