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


class GatherNdConfig(APIConfig):
    def __init__(self):
        super(GatherNdConfig, self).__init__('gather_nd')
        self.run_torch = False
        print("[WARNING]: Pytorch dosen`t support gather_nd currently.")

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(GatherNdConfig, self).init_from_json(filename, config_id,
                                                   unknown_dim)
        self.feed_spec = [
            {
                "range": [0, 10]
            },  # input
            {
                "range": [1, int(min(self.input_shape))]
            }  # index
        ]


class PDGatherNd(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(
            name='x', shape=config.input_shape, dtype=config.input_dtype)
        index = self.variable(
            name='index', shape=config.index_shape, dtype=config.index_dtype)
        # stop_gradient=True)
        result = paddle.gather_nd(x=x, index=index)

        self.feed_list = [x, index]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TorchGatherNd(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(
            name='x', shape=config.input_shape, dtype=config.input_dtype)


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDGatherNd(),
        torch_obj=TorchGatherNd(),
        config=GatherNdConfig())
