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


class OneHotConfig(object):
    def __init__(self, input_shape, depth):
        self.input_shape = input_shape
        self.depth = depth
        self.feed_spec = { "range": [0, depth] }


config = OneHotConfig(input_shape=[32, 128], depth=10)


class PDOneHot(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False, dtype=None):
        import paddle.fluid as fluid

        self.name = "one_hot"
        with fluid.program_guard(self.main_program, self.startup_program):
            input = fluid.data(
                name='input', shape=config.input_shape, dtype='int32', lod_level=0)
            input.stop_gradient = False
            result = fluid.one_hot(input=input, depth=config.depth)

            self.feed_vars = [input]
            self.fetch_vars = [result]


class TFOneHot(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False, dtype=None):
        import tensorflow as tf

        self.name = "one_hot"
        self.allow_growth = True

        input = tf.placeholder(name='input', shape=config.input_shape, dtype=tf.int32)
        result = tf.one_hot(indices=input,
                            depth=config.depth,
                            on_value=None,
                            off_value=None,
                            axis=None,
                            dtype=None)

        self.feed_list = [input]
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(PDOneHot(), TFOneHot(), feed_spec=config.feed_spec)
