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


@benchmark_registry.register("reduce")
class ReduceConfig(APIConfig):
    def __init__(self):
        super(ReduceConfig, self).__init__('reduce')
        self.feed_spec = {"range": [-1, 1]}
        self.api_name = 'sum'
        self.api_list = {'sum': 'sum', 'mean': 'mean'}
        # TODO(Xreki): the api is different in tf.

    #        self.api_list = {
    #            'max': 'reduce_max',
    #            'mean': 'reduce_mean',
    #            'min': 'reduce_min',
    #            'sum': 'reduce_sum',
    #            'prod': 'reduce_prod'
    #        }

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(ReduceConfig, self).init_from_json(filename, config_id,
                                                 unknown_dim)
        if self.axis == None:
            self.axis = []


@benchmark_registry.register("reduce")
class PaddleReduce(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = self.layers(
            config.api_name, x=x, axis=config.axis, keepdim=config.keepdim)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])

    def compute_flop_and_byte(self, config):
        x_shape = config.x_shape
        out_shape = self.fetch_list[0].shape
        forward_flop = numel(x_shape)
        forward_byte = (
            numel(x_shape) + numel(out_shape)) * sizeof(config.x_dtype)
        if not config.backward:
            return forward_flop, forward_byte
        else:
            # To be implemented.
            return None, None


@benchmark_registry.register("reduce")
class TorchReduce(PytorchOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        if len(config.axis) == 0:
            result = self.layers(config.api_name, input=x)
        else:
            result = self.layers(
                config.api_name,
                input=x,
                dim=config.axis,
                keepdim=config.keepdim)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


@benchmark_registry.register("reduce")
class TFReduce(TensorflowOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = self.layers(
            config.api_name,
            input_tensor=x,
            axis=config.axis,
            keepdims=config.keepdim)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])
