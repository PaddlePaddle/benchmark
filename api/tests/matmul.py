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


class MatmulConfig(object):
    def __init__(self, x_shape, y_shape, transpose_x=False, transpose_y=False):
        self.x_shape = x_shape
        self.y_shape = y_shape
        self.transpose_x = transpose_x
        self.transpose_y = transpose_y


config = MatmulConfig(x_shape=[32, 128, 768],
                      y_shape=[32, 768, 256])


class PDMatmul(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False, dtype=None):
        import paddle.fluid as fluid

        self.name = "matmul"
        with fluid.program_guard(self.main_program, self.startup_program):
            x = fluid.data(
                name='x', shape=config.x_shape, dtype='float32', lod_level=0)
            y = fluid.data(
                name='y', shape=config.y_shape, dtype='float32', lod_level=0)
            x.stop_gradient = False
            y.stop_gradient = False
            result = fluid.layers.matmul(x=x,
                                         y=y,
                                         transpose_x=config.transpose_x,
                                         transpose_y=config.transpose_y,
                                         alpha=1.0)

            self.feed_vars = [x, y]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [x, y])


class TFMatmul(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False):
        import tensorflow as tf

        self.name = "matmul"
        self.allow_growth = True

        x = tf.placeholder(name='x', shape=config.x_shape, dtype=tf.float32)
        y = tf.placeholder(name='y', shape=config.y_shape, dtype=tf.float32)
        result = tf.matmul(a=x,
                           b=y,
                           transpose_a=config.transpose_x,
                           transpose_b=config.transpose_y,
                           adjoint_a=False,
                           adjoint_b=False,
                           a_is_sparse=False,
                           b_is_sparse=False)

        self.feed_list = [x, y]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [x, y])


if __name__ == '__main__':
    test_main(PDMatmul(), TFMatmul(), feed_spec=None)
