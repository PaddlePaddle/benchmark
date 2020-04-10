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


class EmbeddingConfig(object):
    def __init__(self, input_shape, table_shape):
        self.input_shape = input_shape
        self.table_shape = table_shape
        self.feed_spec = [{ "range": [0, table_shape[0]] }, # input
                          { "range": [0, 1] }  # table
                         ]
        

config = EmbeddingConfig(input_shape=[10, 10],
                         table_shape=[64, 128])


class PDEmbedding(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False, dtype=None):
        import paddle.fluid as fluid

        self.name = "embedding"
        with fluid.program_guard(self.main_program, self.startup_program):
            input = fluid.data(
                name='input', shape=config.input_shape, dtype='int64', lod_level=0)
            table = fluid.layers.create_parameter(
                shape=config.table_shape, dtype='float32', name='table')
            result = fluid.embedding(input=input,
                                     size=config.table_shape,
                                     param_attr='table')

            self.feed_vars = [input, table]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [table])


class TFEmbedding(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False):
        import tensorflow as tf

        self.name = "embedding"
        self.allow_growth = True

        input = tf.placeholder(name='input', shape=config.input_shape, dtype=tf.int64)
        table = tf.placeholder(name='table', shape=config.table_shape, dtype=tf.float32)
        result = tf.nn.embedding_lookup(ids=input, params=table, max_norm=None)

        self.feed_list = [input, table]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [table])


if __name__ == '__main__':
    test_main(PDEmbedding(), TFEmbedding(), feed_spec=config.feed_spec)
