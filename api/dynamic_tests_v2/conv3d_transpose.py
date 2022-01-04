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


class Conv3dTransposeConfig(APIConfig):
    def __init__(self, op_type="conv3d_transpose"):
        super(Conv3dTransposeConfig, self).__init__(op_type)
        self.feed_spec = [
            {
                "range": [-1, 1]
            },  # x
            {
                "range": [-1, 1],
            }  # weight
        ]

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(Conv3dTransposeConfig, self).init_from_json(filename, config_id,
                                                          unknown_dim)
        if isinstance(self.padding, int):
            self.padding = [self.padding, self.padding, self.padding]
        if self.data_format == "NCDHW":
            self.num_channels = self.input_shape[1]
        elif self.data_format == "NDHWC":
            self.num_channels = self.input_shape[4]
        if isinstance(self.filter_size, int):
            self.filter_size = [
                self.filter_size, self.filter_size, self.filter_size
            ]
        if self.groups is None:
            self.groups = 1
        if self.num_channels % self.groups != 0:
            raise ValueError(
                "the channel of input must be divisible by groups,"
                "received: the channel of input is {}, the shape of input is {}"
                ", the groups is {}".format(self.num_channels,
                                            self.input_shape, self.groups))
        self.filter_shape = [
            self.num_channels // self.groups, self.num_filters,
            self.filter_size[0], self.filter_size[1], self.filter_size[2]
        ]


class PDConv3dTranspose(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(
            name='x', shape=config.input_shape, dtype=config.input_dtype)
        weight = self.variable(
            name='weight', shape=config.filter_shape, dtype=config.input_dtype)
        result = paddle.nn.functional.conv3d_transpose(
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


class TorchConv3dTranspose(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(
            name='x', shape=config.input_shape, dtype=config.input_dtype)
        weight = self.variable(
            name='weight', shape=config.filter_shape, dtype=config.input_dtype)
        result = torch.nn.functional.conv_transpose3d(
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


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDConv3dTranspose(),
        torch_obj=TorchConv3dTranspose(),
        config=Conv3dTransposeConfig())
