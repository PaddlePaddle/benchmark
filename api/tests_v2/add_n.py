#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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


class PDAddN(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        inputs = []
        for i in range(len(config.inputs_shape)):
            input_i = self.variable(
                name='input_' + str(i),
                shape=config.inputs_shape[i],
                dtype=config.inputs_dtype[i])
            inputs.append(input_i)
        result = paddle.add_n(inputs=inputs)

        self.feed_vars = inputs
        self.fetch_vars = [result]


class TFAddN(TensorflowAPIBenchmarkBase):
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


if __name__ == '__main__':
    test_main(PDAddN(), TFAddN(), config=APIConfig("add_n"))
