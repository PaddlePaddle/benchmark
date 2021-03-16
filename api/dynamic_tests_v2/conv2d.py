xw  #   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


class Conv2dConfig(APIConfig):
    def __init__(self, op_type="conv2d"):
        super(Conv2dConfig, self).__init__(op_type)
        self.feed_spec = [
            {
                "range": [-1, 1]
            },  # input
            {
                "range": [-1, 1],
            }  # filters
        ]

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(Conv2dConfig, self).init_from_json(filename, config_id,
                                                 unknown_dim)
        if isinstance(self.padding, int):
            self.padding = [self.padding, self.padding]
        if isinstance(self.dilation, int):
            self.dilation = [self.dilation, self.dilation]
        if self.groups is None:
            self.groups = 1
        assert self.get_in_channels(
        ) % self.groups == 0, "The channel of input must be divisible by groups. "\
            "But received: the channel of input is {}, the shape of input is {}, the groups is {}".format(
            self.get_in_channels(), self.x_shape, self.groups)
        if self.data_format == 'NHWC':
            print(
                "Warning:\n"
                "  1. PyTorch does not have data_format param, it only support NHWC format.\n"
            )
            self.run_torch = False

    def get_in_channels(self):
        return self.x_shape[1] if self.data_format == "NCHW" else self.x_shape[
            3]

    def get_out_channels(self):
        return self.weight_shape[0]


class PDConv2d(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name='weight',
            shape=config.weight_shape,
            dtype=config.weight_dtype)
        result = paddle.nn.functional.conv2d(
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


class TorchConv2d(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name='weight', shape=config.weight_shape, dtype=config.x_dtype)
        result = torch.nn.functional.conv2d(
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
        pd_dy_obj=PDConv2d(), torch_obj=TorchConv2d(), config=Conv2dConfig())
