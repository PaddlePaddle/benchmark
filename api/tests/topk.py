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


class PDTopK(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False, dtype=None):
        import paddle.fluid as fluid

        self.name = "topk"
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data', shape=[16, 1000], dtype='float32', lod_level=0)
            data.stop_gradient = False

            value, indices = fluid.layers.topk(input=data, k=5)

            self.feed_vars = [data]
            self.fetch_vars = [value, indices]
            if backward:
                self.append_gradients([value, indices], [data])


class TFTopK(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False, dtype=None):
        import tensorflow as tf

        self.name = "topk"
        self.allow_growth = True

        data = tf.placeholder(name='data', shape=[16, 1000], dtype=tf.float32)
        value, indices = tf.math.top_k(input=data, k=5)

        self.feed_list = [data]
        self.fetch_list = [value, indices]
        if backward:
            self.append_gradients([value, indices], [data])


if __name__ == '__main__':
    test_main(PDTopK(), TFTopK(), feed_spec=None)
