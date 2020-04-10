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


class PDConcat(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False, dtype=None):
        import paddle.fluid as fluid

        self.name = "concat"
        with fluid.program_guard(self.main_program, self.startup_program):
            data1 = fluid.data(
                name='data1', shape=[100, 200], dtype='float32', lod_level=0)
            data2 = fluid.data(
                name='data2', shape=[100, 200], dtype='float32', lod_level=0)
            data1.stop_gradient = False
            data2.stop_gradient = False
            result = fluid.layers.concat([data1, data2], axis=0)

            self.feed_vars = [data1, data2]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [data1, data2])


class TFConcat(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False):
        import tensorflow as tf

        self.name = "concat"
        self.allow_growth = True

        data1 = tf.placeholder(name='data1', shape=[100, 200], dtype=tf.float32)
        data2 = tf.placeholder(name='data2', shape=[100, 200], dtype=tf.float32)
        result = tf.concat([data1, data2], 0)

        self.feed_list = [data1, data2]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [data1, data2])


if __name__ == '__main__':
    test_main(PDConcat(), TFConcat(), feed_spec=None)
