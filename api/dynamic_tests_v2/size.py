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


class PaddleSize(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(name='input', shape=config.input_shape, dtype=config.input_dtype)
        result = paddle.fluid.layers.size(input)

        self.feed_list = [input]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


class TorchSize(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(name='input', shape=config.input_shape, dtype=config.input_dtype)
        result = torch.numel(input=input)

        self.feed_list = [input]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PaddleSize(),
        torch_obj=TorchSize(),
        config=APIConfig('size'))
