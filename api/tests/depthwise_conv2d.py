#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from conv2d import Conv2dConfig, PDConv2d


class DepthwiseConv2dConfig(Conv2dConfig):
    def __init__(self):
        super(DepthwiseConv2dConfig, self).__init__("depthwise_conv2d")
        self.run_tf = False

    def init_from_json(self, filename, config_id=0):
        super(DepthwiseConv2dConfig, self).init_from_json(filename, config_id)
        assert self.num_channels == self.groups and self.num_filters % self.num_channels == 0
        if isinstance(self.dilation, int):
            self.dilation = [self.dilation, self.dilation]

    def to_tensorflow(self):
        tf_config = super(DepthwiseConv2dConfig, self).to_tensorflow()
        tf_config.filter_shape = [
            self.filter_size[0], self.filter_size[1], self.num_channels,
            self.num_filters // self.groups
        ]
        if isinstance(self.stride, int):
            if self.data_format == "NCHW":
                tf_config.stride = [1, 1, self.stride, self.stride]
            elif self.data_format == "NHWC":
                tf_config.stride = [1, self.stride, self.stride, 1]
        return tf_config

    def _convert_padding(self, padding):
        if isinstance(padding, str):
            return padding

        assert isinstance(padding, list)
        if padding == [1, 1]:
            return "VALID"


class TFDepthwiseConv2d(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        filter = self.variable(
            name='filter', shape=config.filter_shape, dtype=config.input_dtype)
        result = tf.nn.depthwise_conv2d(
            input=input,
            filter=filter,
            strides=config.stride,
            padding=config.padding,
            data_format=config.data_format,
            dilations=config.dilation)
        print(result)

        self.feed_list = [input, filter]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input, filter])


if __name__ == '__main__':
    test_main(PDConv2d(), TFDepthwiseConv2d(), config=DepthwiseConv2dConfig())
