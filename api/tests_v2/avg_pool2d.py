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


class AvgPool2dConfig(APIConfig):
    def __init__(self):
        super(AvgPool2dConfig, self).__init__('avg_pool2d')

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(AvgPool2dConfig, self).init_from_json(filename, config_id,
                                                    unknown_dim)
        if isinstance(self.padding, int):
            self.kernel_size = [self.padding, self.padding]

        # The argument padding of tf's pool2d must be a string and the value
        #   is "SAME" or "VALID".
        if isinstance(self.kernel_size, list):
            if self.kernel_size[0] == self.x_shape[2] and self.kernel_size[
                    1] == self.x_shape[3]:
                self.global_pooling = True
            else:
                self.global_pooling = False

        if not self.global_pooling and not isinstance(
                self.padding, str) and self.padding != [0, 0]:
            self.run_tf = False

    def to_tensorflow(self):
        tf_config = super(AvgPool2dConfig, self).to_tensorflow()
        tf_config.padding = self._convert_padding(self.padding)
        return tf_config

    def _convert_padding(self, padding):
        if isinstance(padding, str):
            return padding

        # It works for current configs, but maybe we need to check whether
        #   pool_size == pool_stride.
        if padding == [0, 0]:
            return "SAME" if self.ceil_mode else "VALID"

        # TODO: fix the call of tf for other padding


class PDAvgPool2d(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = paddle.nn.functional.avg_pool2d(
            x=x,
            kernel_size=config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            ceil_mode=config.ceil_mode,
            data_format=config.data_format)

        self.feed_vars = [x]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TFAvgPool2d(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        if config.global_pooling:
            data_format = "channels_first" if config.data_format == "NCHW" else "channels_last"
            result = tf.keras.layers.GlobalAveragePooling2D(
                data_format=data_format)(x)
        else:
            result = tf.nn.pool(
                input=x,
                window_shape=config.kernel_size,
                pooling_type="AVG",
                strides=config.stride,
                padding=config.padding,
                data_format=config.data_format)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(PDAvgPool2d(), TFAvgPool2d(), config=AvgPool2dConfig())
