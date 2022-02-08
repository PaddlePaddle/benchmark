#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


@benchmark_registry.register("embedding")
class EmbeddingConfig(APIConfig):
    def __init__(self):
        super(EmbeddingConfig, self).__init__('embedding')

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(EmbeddingConfig, self).init_from_json(filename, config_id,
                                                    unknown_dim)
        self.feed_spec = [
            {
                "range": [0, self.weight_shape[0]]
            },  # input
            {
                "range": [0, 1]
            }  # table
        ]
        if self.sparse or self.padding_idx is not None:
            self.run_tf = False


@benchmark_registry.register("embedding")
class PaddleEmbedding(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name="x", shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name='weight',
            shape=config.weight_shape,
            dtype=config.weight_dtype)

        result = paddle.nn.functional.embedding(
            x=x,
            weight=weight,
            sparse=config.sparse,
            padding_idx=config.padding_idx)

        self.feed_list = [x, weight]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [weight])


@benchmark_registry.register("embedding")
class TorchEmbedding(PytorchOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name='weight',
            shape=config.weight_shape,
            dtype=config.weight_dtype)

        result = torch.nn.functional.embedding(
            input=x,
            weight=weight,
            padding_idx=config.padding_idx,
            sparse=config.sparse)

        self.feed_list = [x, weight]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [weight])


@benchmark_registry.register("embedding")
class TFEmbedding(TensorflowOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)

        weight = self.variable(
            name='weight',
            shape=config.weight_shape,
            dtype=config.weight_dtype)

        result = tf.nn.embedding_lookup(ids=x, params=weight, max_norm=None)

        self.feed_list = [x, weight]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [weight])
