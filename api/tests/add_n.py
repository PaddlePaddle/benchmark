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


@benchmark_registry.register("add_n")
class PDAddN(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        inputs = []
        for i in range(len(config.inputs_shape)):
            input_i = self.variable(
                name='input_' + str(i),
                shape=config.inputs_shape[i],
                dtype=config.inputs_dtype[i])
            inputs.append(input_i)
        result = paddle.add_n(inputs=inputs)

        self.feed_list = inputs
        self.fetch_list = [result]


@benchmark_registry.register("add_n")
class TorchAddN(PytorchOpBenchmarkBase):
    def build_graph(self, config):
        inputs = []
        for i in range(len(config.inputs_shape)):
            input_i = self.variable(
                name='input_' + str(i),
                shape=config.inputs_shape[i],
                dtype=config.inputs_dtype[i])
            inputs.append(input_i)
        inputs = torch.stack(inputs, dim=0)
        result = torch.sum(inputs, axis=0)
        self.feed_list = inputs
        self.fetch_list = [result]


@benchmark_registry.register("add_n")
class TFAddN(TensorflowOpBenchmarkBase):
    def build_graph(self, config):
        inputs = []
        for i in range(len(config.inputs_shape)):
            input_i = self.variable(
                name='input_' + str(i),
                shape=config.inputs_shape[i],
                dtype=config.inputs_dtype[i])
            inputs.append(input_i)
        result = tf.add_n(inputs=inputs)

        self.feed_list = inputs
        self.fetch_list = [result]
