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


class PDSlice(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False, dtype=None):
        import paddle.fluid as fluid

        self.name = "slice"
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data',
                shape=[10, 10, 100, 100],
                dtype='float32',
                lod_level=0)
            data.stop_gradient = False
            axes = [0, 1, 2, 3]
            starts = [0, 1, 1, 0]
            ends = [3, 8, 50, 30]
            result = fluid.layers.slice(
                data, axes=axes, starts=starts, ends=ends)

            self.feed_vars = [data]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [data])


class TFSlice(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False, dtype=None):
        import tensorflow as tf

        self.name = "slice"
        self.allow_growth = True

        data = tf.placeholder(
            name='data', shape=[10, 10, 100, 100], dtype=tf.float32)
        begin = [0, 1, 1, 0]
        size = [3, 7, 49, 30]
        result = tf.slice(data, begin, size)

        self.feed_list = [data]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [data])


if __name__ == '__main__':
    test_main(PDSlice(), TFSlice(), feed_spec=None)
