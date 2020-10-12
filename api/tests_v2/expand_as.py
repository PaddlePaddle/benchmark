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


class ExpandAsConfig(APIConfig):
    def __init__(self):
        super(ExpandAsConfig, self).__init__('expand_as')

    def to_tensorflow(self):
        tf_config = super(ExpandAsConfig, self).to_tensorflow()
        if len(self.x_shape) == len(self.y_shape) + 1:
            shape = [self.x_shape[0]] + self.y_shape
        else:
            shape = self.y_shape
        assert len(self.x_shape) == len(
            shape
        ), "The length of shape should be equal to the rank of input x."
        tf_config.multiples = []
        for i in range(len(self.x_shape)):
            tf_config.multiples.append(shape[i] // self.x_shape[i])
        return tf_config


class PDExpandAs(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = paddle.zeros(shape=config.y_shape, dtype=config.y_dtype)
        y.stop_gradient = True
        result = paddle.expand_as(x=x, y=y)

        self.feed_vars = [x]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TFExpandAs(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = tf.tile(input=x, multiples=config.multiples)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(PDExpandAs(), TFExpandAs(), config=ExpandAsConfig())
