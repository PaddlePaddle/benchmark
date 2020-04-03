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

      
class PDEmbedding(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False, dtype=None):
        import paddle.fluid as fluid

        self.name = "embedding"
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data', shape=[10, 10], dtype='int64', lod_level=0)
            table = fluid.layers.create_parameter(
                shape=[64, 128], dtype='float32', name='table')
            result = fluid.embedding(input=data, size=[64, 128], param_attr='table')

            self.feed_vars = [data, table]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [table])


class TFEmbedding(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False):
        import tensorflow as tf

        self.name = "embedding"
        self.allow_growth = True

        data = tf.placeholder(name='data', shape=[10, 10], dtype=tf.int64)
        table = tf.placeholder(name='table', shape=[64, 128], dtype=tf.float32)
        result = tf.nn.embedding_lookup(ids=data, params=table, max_norm=None)

        self.feed_list = [data, table]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [table])


if __name__ == '__main__':
    feed_spec = [{ "range": [0, 64] }, # data
                 { "range": [0, 1] }  # table
                ]
    test_main(PDEmbedding(), TFEmbedding(), feed_spec=feed_spec)
