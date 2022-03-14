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


@benchmark_registry.register("conv3d")
class Conv3dConfig(APIConfig):
    def __init__(self, op_type="conv3d"):
        super(Conv3dConfig, self).__init__(op_type)
        self.run_tf = False
        self.feed_spec = [
            {
                "range": [-1, 1]
            },  # x
            {
                "range": [-1, 1],
            }  # weight
        ]


@benchmark_registry.register("conv3d")
class PaddleConv3d(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name='weight',
            shape=config.weight_shape,
            dtype=config.weight_dtype)
        result = paddle.nn.functional.conv3d(
            x=x,
            weight=weight,
            bias=None,
            stride=config.stride,
            padding=config.padding,
            dilation=config.dilation,
            groups=config.groups,
            data_format=config.data_format)

        self.feed_list = [x, weight]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, weight])


@benchmark_registry.register("conv3d")
class TorchConv3d(PytorchOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name='weight',
            shape=config.weight_shape,
            dtype=config.weight_dtype)
        result = torch.nn.functional.conv3d(
            input=x,
            weight=weight,
            bias=None,
            stride=config.stride,
            padding=config.padding,
            dilation=config.dilation,
            groups=config.groups)

        self.feed_list = [x, weight]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, weight])
