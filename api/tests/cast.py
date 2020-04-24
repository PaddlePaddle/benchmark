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


class CastConfig(object):
    def __init__(self, input_shape, dtype):
        self.input_shape = input_shape
        self.dtype = dtype
        self.feed_spec = {"range": [-10, 10]}


config = CastConfig(input_shape=[10, 10, 100, 100], dtype="float16")


class PDCast(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False, dtype=None):
        import paddle.fluid as fluid

        self.name = "cast"
        with fluid.program_guard(self.main_program, self.startup_program):
            input = fluid.data(
                name='input',
                shape=config.input_shape,
                dtype='float32',
                lod_level=0)
            input.stop_gradient = False
            result = fluid.layers.cast(input, dtype=config.dtype)

            self.feed_vars = [input]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [input])


class TFCast(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False, dtype=None):
        import tensorflow as tf

        self.name = "cast"
        self.allow_growth = True

        input = tf.placeholder(
            name='input', shape=config.input_shape, dtype=tf.float32)
        result = tf.cast(input, dtype=config.dtype)

        self.feed_list = [input]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    test_main(PDCast(), TFCast(), feed_spec=config.feed_spec)
