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

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(DepthwiseConv2dConfig, self).init_from_json(filename, config_id,
                                                          unknown_dim)
        assert self.num_channels == self.groups and self.weight_shape[
            0] % self.num_channels == 0
        if isinstance(self.dilation, int):
            self.dilation = [self.dilation, self.dilation]

    def to_tensorflow(self):
        tf_config = super(DepthwiseConv2dConfig, self).to_tensorflow()
        tf_config.weight_shape = [
            self.weight_shape[2], self.weight_shape[3], self.num_channels,
            self.weight_shape[1]
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
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name='weight', shape=config.weight_shape, dtype=config.x_dtype)
        result = tf.nn.depthwise_conv2d(
            input=x,
            filter=weight,
            strides=config.stride,
            padding=config.padding,
            data_format=config.data_format,
            dilations=config.dilation)

        self.feed_list = [x, weight]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, weight])


if __name__ == '__main__':
    test_main(PDConv2d(), TFDepthwiseConv2d(), config=DepthwiseConv2dConfig())
