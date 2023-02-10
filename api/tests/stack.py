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


@benchmark_registry.register("stack")
class PaddleStack(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        xs = []
        for i in range(len(config.x_shape)):
            x_i = self.variable(
                name='x_' + str(i),
                shape=config.x_shape[i],
                dtype=config.x_dtype[i])
            xs.append(x_i)
        result = paddle.stack(x=xs, axis=config.axis)

        self.feed_list = xs
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, xs)

    def compute_flop_and_byte(self, config):
        num_xs = len(config.x_shape)
        forward_flop = 0
        forward_byte = num_xs * numel(config.x_shape[0]) * sizeof(
            config.x_dtype[0]) * 2
        if not config.backward:
            return forward_flop, forward_byte
        else:
            # To be implemented.
            return None, None


@benchmark_registry.register("stack")
class TorchStack(PytorchOpBenchmarkBase):
    def build_graph(self, config):
        values = []
        for i in range(len(config.x_shape)):
            value_i = self.variable(
                name='value_' + str(i),
                shape=config.x_shape[i],
                dtype=config.x_dtype[i])
            values.append(value_i)
        result = torch.stack(tensors=values, dim=config.axis)

        self.feed_list = values
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, values)


@benchmark_registry.register("stack")
class TFStack(TensorflowOpBenchmarkBase):
    def build_graph(self, config):
        values = []
        for i in range(len(config.x_shape)):
            value_i = self.variable(
                name='value_' + str(i),
                shape=config.x_shape[i],
                dtype=config.x_dtype[i])
            values.append(value_i)
        result = tf.stack(values=values, axis=config.axis)

        self.feed_list = values
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, values)
