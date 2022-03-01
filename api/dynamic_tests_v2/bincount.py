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


class PaddleBincount(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        w = self.variable(name='w', shape=config.w_shape, dtype=config.w_dtype)
        result = paddle.bincount(x=x, weights=w, minlength=config.minlength)

        self.feed_list = [x, w]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TorchBincount(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        w = self.variable(name='w', shape=config.w_shape, dtype=config.w_dtype)
        result = torch.bincount(input=x, weights=w, minlength=config.minlength)

        self.feed_list = [x, w]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PaddleBincount(),
        torch_obj=TorchBincount(),
        config=APIConfig('bincount'))
