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
import numpy as np

import sys
sys.path.append("..")
from common import paddle_api_benchmark as paddle_api
from common import tensorflow_api_benchmark as tensorflow_api


class GatherConfig(object):
    def __init__(self, input_shape, index_shape):
        self.input_shape = input_shape
        self.index_shape = index_shape
        self.feed_spec = [
            {
                "range": [-1, 1]
            },  # input
            {
                "range": [0, input_shape[0]]
            }  # index
        ]


config = GatherConfig(input_shape=[10, 10, 100, 100], index_shape=[4])


class PDGather(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False, dtype=None):
        import paddle.fluid as fluid

        self.name = "gather"
        with fluid.program_guard(self.main_program, self.startup_program):
            input = fluid.data(
                name='input',
                shape=config.input_shape,
                dtype='float32',
                lod_level=0)
            index = fluid.data(
                name='index',
                shape=config.index_shape,
                dtype='int32',
                lod_level=0)
            input.stop_gradient = False
            result = fluid.layers.gather(input=input, index=index)

            self.feed_vars = [input, index]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [input])


class TFGather(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False, dtype=None):
        import tensorflow as tf

        self.name = "gather"
        self.allow_growth = True

        input = tf.placeholder(
            name='input', shape=config.input_shape, dtype=tf.float32)
        index = tf.placeholder(
            name='index', shape=config.index_shape, dtype=tf.int32)
        result = tf.gather(params=input, indices=index)

        self.feed_list = [input, index]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [input, index])


if __name__ == '__main__':
    test_main(PDGather(), TFGather(), feed_spec=config.feed_spec)
