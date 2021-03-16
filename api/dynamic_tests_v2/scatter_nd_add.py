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


class ScatterNdAddConfig(APIConfig):
    def __init__(self):
        super(ScatterNdAddConfig, self).__init__('scatter_nd_add')

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(ScatterNdAddConfig, self).init_from_json(filename, config_id,
                                                       unknown_dim)
        self.feed_spec = [
            {
                "range": [-1, 1]
            },  # input
            {
                "range": [0, int(min(self.input_shape))]
            },  # index
            {
                "range": [-1, 1]
            }  # update
        ]


class PDScatterNdAdd(PaddleAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(
            name='x', shape=config.input_shape, dtype=config.input_dtype)
        index = self.variable(
            name='index',
            shape=config.index_shape,
            dtype=config.index_dtype,
            stop_gradient=True)
        updates = self.variable(
            name='updates',
            shape=config.updates_shape,
            dtype=config.updates_dtype)
        result = paddle.scatter_nd_add(x=x, index=index, updates=updates)

        self.feed_vars = [x, index, updates]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x, updates])


class TorchScatterNdAdd(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        params = self.variable(
            name='params', shape=config.input_shape, dtype=config.input_dtype)
        indices = self.variable(
            name='indices', shape=config.index_shape, dtype=config.index_dtype)
        updates = self.variable(
            name='updates',
            shape=config.updates_shape,
            dtype=config.updates_dtype)
        result = torch.scatter_add(
            src=params, indices=indices, updates=updates)

        self.feed_list = [params, indices, updates]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [params, updates])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDScatterNdAdd(),
        torch_obj=TorchScatterNdAdd(),
        config=ScatterNdAddConfig())
