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


class LinearConfig(APIConfig):
    def __init__(self):
        super(LinearConfig, self).__init__('linear')
        self.feed_spec = [
            {
                "range": [-1, 1]
            },  # x
            {
                "range": [-1, 1],
                "permute": [1, 0]
            },  # weight
            {
                "range": [-1, 1]
            }  # bias
        ]

    def to_pytorch(self):
        torch_config = super(LinearConfig, self).to_pytorch()
        torch_config.weight_shape = [
            self.weight_shape[1], self.weight_shape[0]
        ]
        return torch_config


class PDLinear(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name="x", shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name="weight",
            shape=config.weight_shape,
            dtype=config.weight_dtype)
        if hasattr(config, bias) and config.bias is None:
            bias = None
        else:
            bias = self.variable(
                name="bias", shape=config.bias_shape, dtype=config.bias_dtype)
        result = paddle.nn.functional.linear(x=x, weight=weight, bias=bias)

        self.feed_list = [x, weight, bias]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, weight, bias])


class TorchLinear(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name="x", shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name="weight",
            shape=config.weight_shape,
            dtype=config.weight_dtype)
        if hasattr(config, bias) and config.bias is None:
            bias = None
        else:
            bias = self.variable(
                name="bias", shape=config.bias_shape, dtype=config.bias_dtype)
        result = torch.nn.functional.linear(input=x, weight=weight, bias=bias)

        self.feed_list = [x, weight, bias]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, weight, bias])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDLinear(), torch_obj=TorchLinear(), config=LinearConfig())
