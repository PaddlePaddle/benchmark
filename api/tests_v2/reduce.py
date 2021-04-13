#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


class ReduceConfig(APIConfig):
    def __init__(self):
        super(ReduceConfig, self).__init__('reduce')
        self.feed_spec = {"range": [-1, 1]}
        self.api_name = 'mean'
        self.api_list = {
            'any': 'reduce_any',
            'max': 'reduce_max',
            'mean': 'reduce_mean',
            'min': 'reduce_min',
            'sum': 'reduce_sum',
            'prod': 'reduce_prod'
        }


class PDReduce(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        data = self.variable(
            name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = self.layers(
            config.api_name, x=data, axis=config.axis, keepdim=config.keepdim)

        self.feed_vars = [data]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [data])


class TFReduce(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        data = self.variable(
            name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = self.layers(
            config.api_name,
            input_tensor=data,
            axis=config.axis,
            keepdims=config.keepdim)

        self.feed_list = [data]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [data])


if __name__ == '__main__':
    test_main(PDReduce(), TFReduce(), config=ReduceConfig())
