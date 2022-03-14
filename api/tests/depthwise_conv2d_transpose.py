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


@benchmark_registry.register("depthwise_conv2d_transpose")
class DepthwiseConv2dTransposeConfig(APIConfig):
    def __init__(self):
        super(DepthwiseConv2dTransposeConfig,
              self).__init__("depthwise_conv2d_transpose")
        self.run_tf = False
        self.run_torch = False
        self.feed_spec = [
            {
                "range": [-1, 1]
            },  # input
            {
                "range": [-1, 1],
            }  # filters
        ]

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(DepthwiseConv2dTransposeConfig, self).init_from_json(
            filename, config_id, unknown_dim)
        if isinstance(self.padding, int):
            self.padding = [self.padding, self.padding]
        if self.data_format == "NCHW":
            self.num_channels = self.input_shape[1]
        elif self.data_format == "NHWC":
            self.num_channels = self.input_shape[3]
        if isinstance(self.filter_size, int):
            self.filter_size = [self.filter_size, self.filter_size]

        assert self.num_filters == self.groups and self.num_channels != 1

        self.filter_shape = [
            self.num_channels, self.num_filters // self.groups,
            self.filter_size[0], self.filter_size[1]
        ]


@benchmark_registry.register("depthwise_conv2d_transpose")
class PaddleDepthwiseConv2dTranspose(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(
            name='x', shape=config.input_shape, dtype=config.input_dtype)
        weight = self.variable(
            name='weight', shape=config.filter_shape, dtype=config.input_dtype)

        result = paddle.nn.functional.conv2d_transpose(
            x=x,
            weight=weight,
            output_size=config.output_size,
            stride=config.stride,
            padding=config.padding,
            data_format=config.data_format,
            groups=config.groups,
            dilation=config.dilation)

        self.feed_list = [x, weight]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, weight])
