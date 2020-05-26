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

from common_import import *


class EmbeddingConfig(APIConfig):
    def __init__(self):
        super(EmbeddingConfig, self).__init__('embedding')

    def init_from_json(self, filename, config_id=0):
        super(EmbeddingConfig, self).init_from_json(filename, config_id)
        self.feed_spec = [
            {
                "range": [0, self.size[0]]
            },  # input
            {
                "range": [0, 1]
            }  # table
        ]
        if self.is_sparse or self.padding_idx is not None:
            self.run_tf = False


class PDEmbedding(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            input = fluid.data(
                name='input',
                shape=config.input_shape,
                dtype=config.input_dtype,
                lod_level=0)
            table = fluid.layers.create_parameter(
                shape=config.size, dtype=config.dtype, name='table')
            result = fluid.embedding(
                input=input,
                size=config.size,
                is_sparse=config.is_sparse,
                padding_idx=config.padding_idx,
                param_attr='table',
                dtype=config.dtype)

            self.feed_vars = [input, table]
            self.fetch_vars = [result]
            if config.backward:
                self.append_gradients(result, [table])


class TFEmbedding(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.placeholder(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        table = self.placeholder(
            name='table', shape=config.size, dtype=config.dtype)
        result = tf.nn.embedding_lookup(ids=input, params=table, max_norm=None)

        self.feed_list = [input, table]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [table])


def register_api():
    REGISTER_API_INFO['embedding'] = ['embedding', 'embedding.json']


if __name__ == '__main__':
    test_main(PDEmbedding(), TFEmbedding(), config=EmbeddingConfig())
