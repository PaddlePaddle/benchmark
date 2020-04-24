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

from main import test_main

import sys
sys.path.append("..")
from common import paddle_api_benchmark as paddle_api
from common import tensorflow_api_benchmark as tensorflow_api


class ElementwiseAddConfig(object):
    def __init__(self, x_shape, y_shape):
        self.x_shape = x_shape
        self.y_shape = y_shape


config = ElementwiseAddConfig(x_shape=[1, 32, 1, 768], y_shape=[32, 1, 768, 1])


class PDElementwiseAdd(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False, dtype=None):
        import paddle.fluid as fluid

        self.name = "elementwise_add"
        with fluid.program_guard(self.main_program, self.startup_program):
            x = fluid.data(
                name='x', shape=config.x_shape, dtype='float32', lod_level=0)
            y = fluid.data(
                name='y', shape=config.y_shape, dtype='float32', lod_level=0)
            x.stop_gradient = False
            y.stop_gradient = False
            result = fluid.layers.elementwise_add(x=x, y=y)

            self.feed_vars = [x, y]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [x, y])


class TFElementwiseAdd(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False, dtype=None):
        import tensorflow as tf

        self.name = "elementwise_add"
        self.allow_growth = True

        x = tf.placeholder(name='x', shape=config.x_shape, dtype=tf.float32)
        y = tf.placeholder(name='y', shape=config.y_shape, dtype=tf.float32)
        result = tf.add(x=x, y=y)

        self.feed_list = [x, y]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [x, y])


if __name__ == '__main__':
    test_main(PDElementwiseAdd(), TFElementwiseAdd(), feed_spec=None)
