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


class ReshapeConfig(object):
    def __init__(self, x_shape, shape, shape_is_tensor=False, shape_is_tensor_list=False):
        self.x_shape = x_shape
        self.shape = shape
        self.shape_is_variable = shape_is_tensor
        self.shape_is_tensor_list = shape_is_tensor_list


config = ReshapeConfig(x_shape=[10, 32, 8, 1024],
                       shape=[10, 32, 32, 256])


class PDReshape(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False, dtype=None):
        import paddle.fluid as fluid

        self.name = "reshape"
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data', shape=config.x_shape, dtype='float32', lod_level=0)
            data.stop_gradient = False
            result = fluid.layers.reshape(x=data, shape=config.shape)

            self.feed_vars = [data]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [data])


class TFReshape(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False, dtype=None):
        import tensorflow as tf

        self.name = "reshape"
        self.allow_growth = True

        data = tf.placeholder(name='data', shape=config.x_shape, dtype=tf.float32)
        result = tf.reshape(tensor=data, shape=config.shape)

        self.feed_list = [data]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [data])


if __name__ == '__main__':
    test_main(PDReshape(), TFReshape(), feed_spec=None)
