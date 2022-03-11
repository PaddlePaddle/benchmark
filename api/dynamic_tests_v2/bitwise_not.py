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


class BitwiseNotConfig(APIConfig):
    def __init__(self):
        super(BitwiseNotConfig, self).__init__('bitwise_not')
        self.feed_spec = [{"range": [-100, 100]}]
        # bitwise_not belongs to bitwise op series which only has one parameter
        # thus bitwise_not can reuse bitwise.json. 
        self.alias_name = "bitwise"


class PaddleBitwiseNot(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = paddle.bitwise_not(x=x)

        self.feed_list = [x]
        self.fetch_list = [result]


class TorchBitwiseNot(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = torch.bitwise_not(input=x)

        self.feed_list = [x]
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PaddleBitwiseNot(),
        torch_obj=TorchBitwiseNot(),
        config=BitwiseNotConfig())
