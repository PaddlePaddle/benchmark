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


class GatherConfig(APIConfig):
    def __init__(self):
        super(GatherConfig, self).__init__('gather')

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(GatherConfig, self).init_from_json(filename, config_id,
                                                 unknown_dim)
        self.feed_spec = [
            {
                "range": [8, 10]
            },  # input
            {
                "range": [1, self.input_shape[self.axis]]
            }  # index
        ]

        if self.input_shape != self.index_shape:
            self.input_shape = self.index_shape


class PDGather(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(
            name='x', shape=config.input_shape, dtype=config.input_dtype)
        index = self.variable(
            name='index',
            shape=config.index_shape,
            dtype=config.index_dtype,
            stop_gradient=True)
        result = paddle.gather(x=x, index=index, axis=config.axis)

        self.feed_list = [x, index]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TorchGather(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(
            name='x', shape=config.input_shape, dtype=config.input_dtype)
        index = self.variable(
            name='index', shape=config.index_shape, dtype="int64")
        result = torch.gather(input=x, index=index, dim=config.axis)

        self.feed_list = [x, index]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDGather(), torch_obj=TorchGather(), config=GatherConfig())
