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

from main import test_main

import sys
sys.path.append("..")
from common import paddle_api_benchmark as paddle_api
from common import tensorflow_api_benchmark as tensorflow_api


class Pool2dConfig(object):
    def __init__(self,
                 input_shape,
                 pool_size,
                 pool_type="max",
                 stride=[1, 1],
                 padding=[0, 0]):
        self.input_shape = input_shape
        self.pool_size = pool_size
        self._pool_type = pool_type
        self.pool_stride = stride
        self._padding = padding
        self.data_format = "NCHW"
        self.use_cudnn = True

    def pool_type(self, for_tensorflow=False):
        if not for_tensorflow:
            return self._pool_type.lower()
        else:
            return self._pool_type.upper()

    def pool_padding(self, for_tensorflow=False):
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


config = Pool2dConfig(
    input_shape=[10, 10, 100, 100],
    pool_size=[3, 3],
    pool_type="avg",
    stride=[3, 3],
    padding="SAME")


class PDPool2d(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False, dtype=None):
        import paddle.fluid as fluid

        self.name = "pool2d"
        with fluid.program_guard(self.main_program, self.startup_program):
            input = fluid.data(
                name='input',
                shape=config.input_shape,
                dtype='float32',
                lod_level=0)
            input.stop_gradient = False
            result = fluid.layers.pool2d(
                input=input,
                pool_size=config.pool_size,
                pool_type=config.pool_type(),
                pool_stride=config.pool_stride,
                pool_padding=config.pool_padding(),
                global_pooling=False,
                use_cudnn=config.use_cudnn,
                ceil_mode=False,
                exclusive=True,
                data_format=config.data_format)

            self.feed_vars = [input]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [input])


class TFPool2d(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False, dtype=None):
        import tensorflow as tf

        self.name = "pool2d"
        self.allow_growth = True

        input = tf.placeholder(
            name='input', shape=config.input_shape, dtype=tf.float32)
        result = tf.nn.pool(
            input,
            window_shape=config.pool_size,
            pooling_type=config.pool_type(),
            strides=config.pool_stride,
            padding=config.pool_padding(for_tensorflow=True),
            data_format=config.data_format)

        self.feed_list = [input]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [input])


def register_api():
    REGISTER_API_INFO['pool2d'] = ['pool2d', 'pool2d.json']


if __name__ == '__main__':
    test_main(PDPool2d(), TFPool2d(), feed_spec=None)
