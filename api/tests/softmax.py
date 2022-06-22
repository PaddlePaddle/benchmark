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


@benchmark_registry.register("softmax")
class SoftmaxConfig(APIConfig):
    def __init__(self):
        super(SoftmaxConfig, self).__init__("softmax")
        self.feed_spec = {"range": [-1, 1]}


@benchmark_registry.register("softmax")
class PaddleSoftmax(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name="x", shape=config.x_shape, dtype=config.x_dtype)
        result = paddle.nn.functional.softmax(x=x, axis=config.axis)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])

    def compute_flop_and_byte(self, config):
        x_shape = config.x_shape
        # (exp(x[i, j]) - max(x[i, j])) / sum(exp(x[i, j]) - max(x[i, j]))
        forward_flop = numel(x_shape) * 5
        forward_byte = numel(x_shape) * 2 * sizeof(config.x_dtype)
        if not config.backward:
            return forward_flop, forward_byte
        else:
            # To be implemented.
            return None, None


@benchmark_registry.register("softmax")
class TorchSoftmax(PytorchOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = torch.nn.functional.softmax(input=x, dim=config.axis)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


@benchmark_registry.register("softmax")
class TFSoftmax(TensorflowOpBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name='input', shape=config.x_shape, dtype=config.x_dtype)
        result = tf.nn.softmax(logits=input, axis=config.axis)

        self.feed_list = [input]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])
