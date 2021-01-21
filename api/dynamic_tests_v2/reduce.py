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
        self.api_name = 'sum'
        self.api_list = {'sum': 'sum'}

    def init_from_json(self, filename, config_id=3, unknown_dim=16):
        super(ReduceConfig, self).init_from_json(filename, config_id,
                                                 unknown_dim)
        if self.axis == None:
            self.run_torch = False


class PDReduce(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = self.layers(
            config.api_name, x=x, axis=config.axis, keepdim=config.keepdim)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TorchReduce(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = self.layers(
            config.api_name, input=x, dim=config.axis, keepdim=config.keepdim)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDReduce(), torch_obj=TorchReduce(), config=ReduceConfig())
