#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


class SoftmaxConfig(APIConfig):
    def __init__(self):
        super(SoftmaxConfig, self).__init__("softmax")
        self.feed_spec = {"range": [-1, 1]}


class PDSoftmax(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        input = self.variable(
            name='input', shape=config.x_shape, dtype=config.x_dtype)
        result = paddle.nn.functional.softmax(x=input, axis=config.axis)

        self.feed_vars = [input]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [input])


class TFSoftmax(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name='input', shape=config.x_shape, dtype=config.x_dtype)
        result = tf.nn.softmax(logits=input, axis=config.axis)

        self.feed_list = [input]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    test_main(PDSoftmax(), TFSoftmax(), config=SoftmaxConfig())
