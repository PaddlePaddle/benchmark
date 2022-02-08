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


@benchmark_registry.register("expand")
class ExpandConfig(APIConfig):
    def __init__(self):
        super(ExpandConfig, self).__init__('expand')

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(ExpandConfig, self).init_from_json(filename, config_id,
                                                 unknown_dim)
        if len(self.x_shape) == len(self.shape) + 1:
            self.shape = [self.x_shape[0]] + self.shape
        assert len(self.shape) == len(
            self.x_shape), "x and shape should have the same length."

    def to_tensorflow(self):
        tf_config = super(ExpandConfig, self).to_tensorflow()
        tf_config.multiples = []
        for i in range(len(self.x_shape)):
            tf_config.multiples.append(self.shape[i] // self.x_shape[i])
        return tf_config


@benchmark_registry.register("expand")
class PaddleExpand(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = paddle.expand(x=x, shape=config.shape)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


@benchmark_registry.register("expand")
class TorchExpand(PytorchOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = x.expand(config.shape)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


@benchmark_registry.register("expand")
class TFExpand(TensorflowOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = tf.tile(input=x, multiples=config.multiples)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])
