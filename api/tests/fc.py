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


class FCConfig(object):
    def __init__(self, input_shape, size, num_flatten_dims=-1):
        row = 1
        col = 1
        if num_flatten_dims < 0:
            num_flatten_dims = num_flatten_dims + len(input_shape)
        for i in range(len(input_shape)):
            if i < num_flatten_dims:
                row = row * input_shape[i]
            else:
                col = col * input_shape[i]
        self.input_shape = [row, col]
        self.size = size


config = FCConfig(input_shape=[16384, 4],
                  size=4)


class PDFC(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False, dtype=None):
        import paddle.fluid as fluid

        self.name = "fc"
        with fluid.program_guard(self.main_program, self.startup_program):
            input = fluid.data(
                name='input', shape=config.input_shape, dtype='float32', lod_level=0)
            input.stop_gradient = False
            result = fluid.layers.fc(
                input=input,
                size=config.size,
                num_flatten_dims=-1,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.ConstantInitializer(0.5)),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.ConstantInitializer(0.1)),
                act=None)

            self.feed_vars = [input]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [input])


class TFFC(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False, dtype=None):
        import tensorflow as tf

        self.name = "fc"
        self.allow_growth = True

        input = tf.placeholder(name='input', shape=config.input_shape, dtype=tf.float32)
        result = tf.contrib.layers.fully_connected(
            inputs=input,
            num_outputs=config.size,
            weights_initializer=tf.constant_initializer(0.5),
            biases_initializer=tf.constant_initializer(0.1),
            activation_fn=None)

        self.feed_list = [input]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    test_main(PDFC(), TFFC(), feed_spec=None)
