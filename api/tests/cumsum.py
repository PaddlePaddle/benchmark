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

from main import test_main

import sys
sys.path.append("..")
from common import paddle_api_benchmark as paddle_api
from common import tensorflow_api_benchmark as tensorflow_api


class CumsumConfig(object):
    def __init__(self, x_shape, axis=-1):
        self.x_shape = x_shape
        self.axis = axis if axis >= 0 else axis + len(x_shape)
        self.exclusive = False
        self.reverse = False


config = CumsumConfig(x_shape=[1700971, 1],
                      axis=0)


class PDCumsum(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False, dtype=None):
        import paddle.fluid as fluid

        self.name = "cumsum"
        with fluid.program_guard(self.main_program, self.startup_program):
            x = fluid.data(
                name='x', shape=config.x_shape, dtype="float32", lod_level=0)
            x.stop_gradient = False
            result = fluid.layers.cumsum(x=x,
                                         axis=config.axis,
                                         exclusive=config.exclusive,
                                         reverse=config.reverse)

            self.feed_vars = [x]
            self.fetch_vars = [result]


class TFCumsum(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False, dtype=None):
        import tensorflow as tf

        self.name = "cumsum"
        self.allow_growth = True

        x = self.placeholder(name='x', shape=config.x_shape, dtype=tf.float32)
        result = tf.cumsum(x=x,
                           axis=config.axis,
                           exclusive=config.exclusive,
                           reverse=config.reverse)

        self.feed_list = [x]
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(PDCumsum(), TFCumsum(), feed_spec=None)
