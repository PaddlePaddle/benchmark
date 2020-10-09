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

        tf_config.paddings = np.zeros([4, 2]).astype("int32")
        offset = 2 if self.data_format == 'NCHW' else 1  # NHWC
        for i in range(2):
            tf_config.paddings[i + offset][0] = self.paddings[2 * i]
            tf_config.paddings[i + offset][1] = self.paddings[2 * i + 1]
        return tf_config


class PDPad2d(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        input = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        result = paddle.nn.functional.pad2d(
            input=input,
            paddings=config.paddings,
            mode=config.mode,
            pad_value=config.pad_value,
            data_format=config.data_format)

        self.feed_vars = [input]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [input])


class TFPad2d(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        paddings = tf.constant(
            shape=config.paddings.shape, dtype=tf.int32, value=config.paddings)
        result = tf.pad(tensor=input,
                        paddings=paddings,
                        mode=config.mode,
                        constant_values=config.pad_value)

        self.feed_list = [input]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    test_main(PDPad2d(), TFPad2d(), config=Pad2dConfig())