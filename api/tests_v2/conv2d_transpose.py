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
from conv2d import Conv2dConfig


class Conv2dTransposeConfig(Conv2dConfig):
    def __init__(self):
        super(Conv2dTransposeConfig, self).__init__("conv2d_transpose")

    def init_from_json(self, filename, config_id=0, unknown_dim=2):
        super(Conv2dTransposeConfig, self).init_from_json(filename, config_id,
                                                          unknown_dim)

        # The argument padding of tf's conv2d_transpose must be a string and
        # the value is "SAME" or "VALID".
        if not isinstance(
                self.padding,
                str) and self.padding != [0, 0] and self.padding != [1, 1]:
            print(
                "Warning:\n"
                "  1. tf is disabled because the argument padding of tf's "
                "conv2d_transpose must be a string and the value is \"SAME\" "
                "or \"VALID\". Please add rule to convert this kind of padding to string.\n"
            )
            self.run_tf = False

    def to_tensorflow(self):
        assert self.output_size is not None
        tf_config = super(Conv2dTransposeConfig, self).to_tensorflow()
        tf_config.weight_shape = [
            self.weight_shape[2], self.weight_shape[3], self.weight_shape[1],
            self.weight_shape[0]
        ]
        if self.data_format == "NCHW":
            tf_config.output_size = [
                self.x_shape[0], self.weight_shape[1], self.output_size[0],
                self.output_size[1]
            ]
        elif self.data_format == "NHWC":
            tf_config.output_size = [
                self.x_shape[0], self.output_size[0], self.output_size[1],
                self.weight_shape[1]
            ]
        tf_config.padding = self._convert_padding(self.padding)
        return tf_config

    def _convert_padding(self, padding):
        if isinstance(padding, str):
            return padding

        assert isinstance(padding, list)

        # It works for current configs, but maybe we need to add some check.
        return "VALID" if padding == [0, 0] else "SAME"


class PDConv2dTranspose(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name="x", shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name="weight",
            shape=config.weight_shape,
            dtype=config.weight_dtype)

        result = paddle.nn.functional.conv2d_transpose(
            x=x,
            weight=weight,
            bias=None,
            stride=config.stride,
            padding=config.padding,
            output_padding=0,
            dilation=config.dilation,
            groups=config.groups,
            output_size=config.output_size,
            data_format=config.data_format)

        self.feed_vars = [x, weight]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x, weight])


class TFConv2dTranspose(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name='input', shape=config.x_shape, dtype=config.x_dtype)
        filter = self.variable(
            name='filter',
            shape=config.weight_shape,
            dtype=config.weight_dtype)
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
            self.append_gradients(result, [input, filter])


if __name__ == '__main__':
    test_main(
        PDConv2dTranspose(),
        TFConv2dTranspose(),
        config=Conv2dTransposeConfig())
