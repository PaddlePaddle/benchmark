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


class PaddleCumsum(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        self.feed_list = [x]

    def run_graph(self, config):
        result = paddle.cumsum(x=self.feed_list[0], axis=config.axis)
        self.fetch_list = [result]


class TorchCumsum(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        self.feed_list = [x]

    def run_graph(self, config):
        result = torch.cumsum(x=self.feed_list[0], axis=config.axis)
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PaddleCumsum(),
        torch_obj=TorchCumsum(),
        config=APIConfig("cumsum"))
