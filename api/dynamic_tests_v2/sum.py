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


class PDSum(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = paddle.sum(x=x, axis=config.axis, keepdim=config.keepdim)

        self.feed_list = [x]
        self.fetch_list = [result]


class TorchSum(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = torch.sum(input=x, dim=config.axis, keepdim=config.keepdim)

        self.feed_list = [x]
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDSum(), torch_obj=TorchSum(), config=APIConfig("sum"))