#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


@benchmark_registry.register("pad2d")
class Pad2dConfig(APIConfig):
    def __init__(self):
        super(Pad2dConfig, self).__init__('pad2d')

    def to_tensorflow(self):
        tf_config = super(Pad2dConfig, self).to_tensorflow()
        if self.mode == 'constant':
            tf_config.mode = 'CONSTANT'
        elif self.mode == 'reflect':
            tf_config.mode = 'REFLECT'
        elif self.mode == 'edge':
            tf_config.mode = 'SYMMETRIC'

        tf_config.pad = np.zeros([4, 2]).astype("int32")
        offset = 2 if self.data_format == 'NCHW' else 1  # NHWC
        for i in range(2):
            tf_config.pad[i + offset][0] = self.pad[2 * i]
            tf_config.pad[i + offset][1] = self.pad[2 * i + 1]
        return tf_config


@benchmark_registry.register("pad2d")
class PaddlePad2d(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = paddle.nn.functional.pad(x=x,
                                          pad=config.pad,
                                          mode=config.mode,
                                          value=config.value,
                                          data_format=config.data_format)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


@benchmark_registry.register("pad2d")
class TorchPad2d(PytorchOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = torch.nn.functional.pad(input=x,
                                         pad=config.pad,
                                         mode=config.mode,
                                         value=config.value)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


@benchmark_registry.register("pad2d")
class TFPad2d(TensorflowOpBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name='input', shape=config.x_shape, dtype=config.x_dtype)
        paddings = tf.constant(
            shape=config.pad.shape, dtype=tf.int32, value=config.pad)
        result = tf.pad(tensor=input,
                        paddings=paddings,
                        mode=config.mode,
                        constant_values=config.value)

        self.feed_list = [input]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])
