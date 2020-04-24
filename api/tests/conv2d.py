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

from main import test_main

import sys
sys.path.append("..")
from common import paddle_api_benchmark as paddle_api
from common import tensorflow_api_benchmark as tensorflow_api


class Conv2dConfig(object):
    def __init__(self,
                 input_shape,
                 num_filters,
                 filter_size,
                 stride=[1, 1],
                 padding=[0, 0]):
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self._padding = padding
        self.dilation = [1, 1]
        self.groups = 1
        self.data_format = "NCHW"
        self.use_cudnn = True
        self.feed_spec = [
            {
                "range": [0, 1]
            },  # input
            {
                "permute": [2, 3, 1, 0]
            }  # filters
        ]

    def filter_shape(self, for_tensorflow=False):
        if self.data_format == "NCHW":
            num_channels = self.input_shape[1]
        elif self.data_format == "NHWC":
            num_channels = self.input_shape[4]
        if not for_tensorflow:
            return [
                self.num_filters, num_channels, self.filter_size[0],
                self.filter_size[1]
            ]
        else:
            return [
                self.filter_size[0], self.filter_size[1], num_channels,
                self.num_filters
            ]

    def padding(self, for_tensorflow=False):
        if not for_tensorflow or isinstance(self._padding, str):
            return self._padding

        assert isinstance(self._padding, list)
        pad_top = self._padding[0] if len(
            self._padding) == 2 else self._padding[0]
        pad_bottom = self._padding[0] if len(
            self._padding) == 2 else self._padding[1]
        pad_left = self._padding[1] if len(
            self._padding) == 2 else self._padding[2]
        pad_right = self._padding[1] if len(
            self._padding) == 2 else self._padding[3]

        if self.data_format == "NCHW":
            return [[0, 0], [0, 0], [pad_top, pad_bottom],
                    [pad_left, pad_right]]
        elif self.data_format == "NHWC":
            return [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right],
                    [0, 0]]


#config = Conv2dConfig(input_shape=[32, 3, 224, 224],
#                      num_filters=64,
#                      filter_size=[7, 7],
#                      stride=[2, 2],
#                      padding="SAME")

config = Conv2dConfig(
    input_shape=[1, 1, 80, 1008],
    num_filters=1,
    filter_size=[3, 32],
    stride=[1, 16],
    padding=[1, 8])


class PDConv2d(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False, dtype=None):
        import paddle.fluid as fluid

        self.name = "conv2d"
        with fluid.program_guard(self.main_program, self.startup_program):
            input = fluid.data(
                name='input',
                shape=config.input_shape,
                dtype=dtype,
                lod_level=0)
            filter = fluid.layers.create_parameter(
                name='filter', shape=config.filter_shape(), dtype=dtype)
            input.stop_gradient = False
            result = fluid.layers.conv2d(
                input=input,
                num_filters=config.num_filters,
                filter_size=config.filter_size,
                stride=config.stride,
                padding=config.padding(),
                dilation=config.dilation,
                groups=config.groups,
                param_attr='filter',
                bias_attr=False,
                use_cudnn=config.use_cudnn,
                act=None,
                data_format=config.data_format)

            self.feed_vars = [input, filter]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [input])


class TFConv2d(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False, dtype=None):
        import tensorflow as tf

        self.name = "conv2d"
        self.allow_growth = True

        input = tf.placeholder(
            name='input', shape=config.input_shape, dtype=tf.float32)
        filter = tf.placeholder(
            name='filter',
            shape=config.filter_shape(for_tensorflow=True),
            dtype=tf.float32)
        result = tf.nn.conv2d(
            input=input,
            filter=filter,
            strides=config.stride,
            padding=config.padding(for_tensorflow=True),
            data_format=config.data_format,
            dilations=config.dilation,
            use_cudnn_on_gpu=config.use_cudnn)

        self.feed_list = [input, filter]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    test_main(PDConv2d(), TFConv2d(), feed_spec=config.feed_spec)
