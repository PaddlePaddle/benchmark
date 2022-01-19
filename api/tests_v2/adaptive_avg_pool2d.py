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
import tensorflow_addons as tfa


class AdaptiveAvgPool2dConfig(APIConfig):
    def __init__(self):
        super(AdaptiveAvgPool2dConfig, self).__init__("adaptive_avg_pool2d")
        self.feed_spec = {"range": [-1, 1]}


class PDAdaptiveAvgPool2D(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        adaptive_avg_pool2d = paddle.nn.AdaptiveAvgPool2D(
            output_size=config.output_size, data_format=config.data_format)
        result = adaptive_avg_pool2d(x)

        self.feed_vars = [x]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TFAdaptiveAvgPool2d(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        data_format = "channels_first"
        if config.data_format == 'NHWC':
            data_format = "channels_last"
        tf_func = tfa.layers.AdaptiveAveragePooling2D(
            output_size=config.output_size, data_format=data_format)
        out = tf_func(x)
        self.feed_list = [x]
        self.fetch_list = [out]
        if config.backward:
            self.append_gradients(out, [x])


if __name__ == '__main__':
    test_main(
        PDAdaptiveAvgPool2D(),
        TFAdaptiveAvgPool2d(),
        config=AdaptiveAvgPool2dConfig())
