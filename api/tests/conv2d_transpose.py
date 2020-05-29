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


class Conv2dTransposeConfig(APIConfig):
    def __init__(self):
        super(Conv2dTransposeConfig, self).__init__('conv2d_transpose')
        self.feed_spec = [
            {
                "range": [0, 1]
            },  # input
            {
                "permute": [2, 3, 1, 0]
            }  # filters
        ]

    def init_from_json(self, filename, config_id=0):
        super(Conv2dTransposeConfig, self).init_from_json(filename, config_id)
        if self.data_format == "NCHW":
            self.num_channels = self.input_shape[1]
        elif self.data_format == "NHWC":
            self.num_channels = self.input_shape[3]
        if self.input_shape[0] == -1:
            self.input_shape[0] = 64
        if isinstance(self.filter_size, int):
            self.filter_size = [self.filter_size, self.filter_size]
        if self.num_channels % self.groups != 0:
            raise ValueError(
                "the channel of input must be divisible by groups,"
                "received: the channel of input is {}, the shape of input is {}"
                ", the groups is {}".format(self.num_channels,
                                            self.input_shape, self.groups))
        self.filter_shape = [
            self.num_channels // self.groups, self.num_filters,
            self.filter_size[0], self.filter_size[1]
        ]
        # The argument padding of tf's conv2d_transpose must be a string and the value is "SAME" or "VALID".
        if self.output_size is None or not isinstance(self.padding, str):
            self.run_tf = False

    def to_tensorflow(self):
        tf_config = super(Conv2dTransposeConfig, self).to_tensorflow()
        tf_config.filter_shape = [
            self.filter_size[0], self.filter_size[1], self.num_filters,
            self.num_channels
        ]
        return tf_config


class PDConv2dTranspose(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        input = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        filter = self.variable(
            name='filter', shape=config.filter_shape, dtype=config.input_dtype)
        result = fluid.layers.conv2d_transpose(
            input=input,
            num_filters=config.num_filters,
            output_size=config.output_size,
            filter_size=config.filter_size,
            padding=config.padding,
            stride=config.stride,
            dilation=config.dilation,
            groups=config.groups,
            param_attr="filter",
            bias_attr=False,
            use_cudnn=config.use_cudnn,
            act=None,
            data_format=config.data_format)
        print(result)

        self.feed_vars = [input, filter]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [input])


class TFConv2dTranspose(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        filter = self.variable(
            name='filter', shape=config.filter_shape, dtype=config.input_dtype)
        result = tf.nn.conv2d_transpose(
            input=input,
            filters=filter,
            output_shape=config.output_size,
            strides=config.stride,
            padding=config.padding,
            data_format=config.data_format,
            dilations=config.dilation)

        self.feed_list = [input, filter]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    test_main(
        PDConv2dTranspose(),
        TFConv2dTranspose(),
        config=Conv2dTransposeConfig())
