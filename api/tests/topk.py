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


@benchmark_registry.register("topk")
class PaddleTopK(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        value, indices = paddle.topk(x=x, k=config.k)

        self.feed_list = [x]
        self.fetch_list = [value, indices]
        #if config.backward:
        #    self.append_gradients([value], [x])


@benchmark_registry.register("topk")
class TorchTopK(PytorchOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        value, indices = torch.topk(input=x, k=config.k)

        self.feed_list = [x]
        self.fetch_list = [value, indices]


@benchmark_registry.register("topk")
class TFTopK(TensorflowOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        value, indices = tf.math.top_k(input=x, k=config.k)

        self.feed_list = [x]
        self.fetch_list = [value, indices]
        if config.backward:
            self.append_gradients([value], [x])
