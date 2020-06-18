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


class Pool2dConfig(APIConfig):
    def __init__(self):
        super(Pool2dConfig, self).__init__('pool2d')

    def init_from_json(self, filename, config_id=0):
        super(Pool2dConfig, self).init_from_json(filename, config_id)
        self.pool_type = self.pool_type.lower()
        # The argument of tf's padding algorithm, must be "SAME" or "VALID".
        if not isinstance(self.pool_padding, str):
            self.run_tf = False

    def to_tensorflow(self):
        tf_config = super(Pool2dConfig, self).to_tensorflow()
        tf_config.pool_type = tf_config.pool_type.upper()
        tf_config.pool_padding = self._convert_padding(self.pool_padding)
        return tf_config

    def _convert_padding(self, padding):
        if isinstance(padding, str):
            return padding
        if isinstance(padding, int):
            padding = [padding, padding]

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


class PDPool2d(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        input = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        result = fluid.layers.pool2d(
            input=input,
            pool_size=config.pool_size,
            pool_type=config.pool_type,
            pool_stride=config.pool_stride,
            pool_padding=config.pool_padding,
            global_pooling=config.global_pooling,
            use_cudnn=config.use_cudnn,
            ceil_mode=config.ceil_mode,
            exclusive=config.exclusive,
            data_format=config.data_format)

        self.feed_vars = [input]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [input])


class TFPool2d(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        result = tf.nn.pool(
            input,
            window_shape=config.pool_size,
            pooling_type=config.pool_type,
            strides=config.pool_stride,
            padding=config.pool_padding,
            data_format=config.data_format)

        self.feed_list = [input]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    test_main(PDPool2d(), TFPool2d(), config=Pool2dConfig())
