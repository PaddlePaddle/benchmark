#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


class IndexSelectConfig(APIConfig):
    def __init__(self):
        super(IndexSelectConfig, self).__init__('index_select')
        self.run_tf = False

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(IndexSelectConfig, self).init_from_json(filename, config_id,
                                                      unknown_dim)
        dim = self.axis
        self.feed_spec = [
            {
                "range": [0, 10]
            },  # x
            {
                "range": [0, self.x_shape[dim]]
            }  # index
        ]


class PDIndexSelect(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        idx = self.variable(
            name='index', shape=config.index_shape, dtype=config.index_dtype)
        result = paddle.index_select(x=x, index=idx, axis=config.axis)

        self.feed_list = [x, idx]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TorchIndexSelect(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        idx = self.variable(
            name='index', shape=config.index_shape, dtype=config.index_dtype)
        result = torch.index_select(input=x, dim=config.axis, index=idx)

        self.feed_list = [x, idx]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDIndexSelect(),
        torch_obj=TorchIndexSelect(),
        config=IndexSelectConfig())
