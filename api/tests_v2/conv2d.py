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


class Conv2dConfig(APIConfig):
    def __init__(self, op_type="conv2d"):
        super(Conv2dConfig, self).__init__(op_type)
        self.feed_spec = [
            {
                "range": [-1, 1]
            },  # input
            {
                "range": [-1, 1],
                "permute": [2, 3, 1, 0]
            }  # filters
        ]

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(Conv2dConfig, self).init_from_json(filename, config_id,
                                                 unknown_dim)
        if use_gpu and self.data_format == "NCHW":
            print(
                "Warning:\n"
                "  1. The tf's conv ops currently only supports the NHWC tensor "
                "format on the CPU. Please add a rule to support it.\n")
            self.run_tf = False

        if isinstance(self.padding, int):
            self.padding = [self.padding, self.padding]
        if self.data_format == "NCHW":
            self.num_channels = self.x_shape[1]
        elif self.data_format == "NHWC":
            self.num_channels = self.x_shape[3]
        if isinstance(self.filter_size, int):
            self.filter_size = [self.filter_size, self.filter_size]
        if self.groups is None:
            self.groups = 1
        if self.num_channels % self.groups != 0:
            raise ValueError(
                "the channel of input must be divisible by groups,"
                "received: the channel of input is {}, the shape of input is {}"
                ", the groups is {}".format(self.num_channels, self.x_shape,
                                            self.groups))
        self.weight_shape = [
            self.num_filters, self.num_channels // self.groups,
            self.filter_size[0], self.filter_size[1]
        ]

    def to_tensorflow(self):
        tf_config = super(Conv2dConfig, self).to_tensorflow()
        tf_config.weight_shape = [
            self.filter_size[0], self.filter_size[1],
            self.num_channels // self.groups, self.num_filters
        ]
        tf_config.padding = self._convert_padding(self.padding)
        return tf_config

    def _convert_padding(self, padding):
        if isinstance(padding, str):
            return padding

        assert isinstance(padding, list)
        assert len(padding) == 2 or len(padding) == 4
        pad_top = padding[0] if len(padding) == 2 else padding[0]
        pad_bottom = padding[0] if len(padding) == 2 else padding[1]
        pad_left = padding[1] if len(padding) == 2 else padding[2]
        pad_right = padding[1] if len(padding) == 2 else padding[3]

        if self.data_format == "NCHW":
            return [[0, 0], [0, 0], [pad_top, pad_bottom],
                    [pad_left, pad_right]]
        elif self.data_format == "NHWC":
            return [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right],
                    [0, 0]]


class PDConv2d(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name='weight', shape=config.weight_shape, dtype=config.x_dtype)
        result = paddle.nn.functional.conv2d(
            x=x,
            weight=weight,
            bias=None,
            stride=config.stride,
            padding=config.padding,
            dilation=config.dilation,
            groups=config.groups,
            data_format=config.data_format)

        self.feed_vars = [x, weight]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x, weight])


class TFConv2d(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name='weight', shape=config.weight_shape, dtype=config.x_dtype)
        result = tf.nn.conv2d(
            input=x,
            filters=weight,
            strides=config.stride,
            padding=config.padding,
            data_format=config.data_format,
            dilations=config.dilation)

        self.feed_list = [x, weight]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, weight])


if __name__ == '__main__':
    test_main(PDConv2d(), TFConv2d(), config=Conv2dConfig())
