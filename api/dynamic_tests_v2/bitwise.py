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


class BitwiseConfig(APIConfig):
    def __init__(self):
        super(BitwiseConfig, self).__init__('bitwise')
        self.api_name = 'bitwise_and'
        self.api_list = {
            'bitwise_and': 'bitwise_and',
            'bitwise_or': 'bitwise_or',
            'bitwise_xor': 'bitwise_xor'
        }
        self.feed_spec = [{"range": [-100, 100]}, {"range": [-100, 100]}]


class PaddleBitwise(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)
        result = self.layers(config.api_name, x=x, y=y)

        self.feed_list = [x, y]
        self.fetch_list = [result]


class TorchBitwise(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)
        result = self.layers(config.api_name, input=x, other=y)

        self.feed_list = [x, y]
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PaddleBitwise(),
        torch_obj=TorchBitwise(),
        config=BitwiseConfig())
