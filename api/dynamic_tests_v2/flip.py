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


class FlipConfig(APIConfig):
    def __init__(self):
        super(FlipConfig, self).__init__("flip")

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(FlipConfig, self).init_from_json(filename, config_id,
                                               unknown_dim)
        self.feed_spec = [
            {
                "range": [-1, 1]
            },  # x
            {
                "range": [0, len(self.x_shape)]
            }  # dims
        ]


class PDFlip(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        dims = []
        dims.append(config.axis)
        result = paddle.flip(x=x, axis=dims)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TorchFlip(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        dims = []
        dims.append(config.axis)
        result = torch.flip(input=x, dims=dims)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, self.feed_list)


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDFlip(), torch_obj=TorchFlip(), config=APIConfig("flip"))
