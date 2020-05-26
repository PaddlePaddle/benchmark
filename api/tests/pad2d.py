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
        self.run_tf = False

    def to_tensorflow(self):
        tf_config = self
        if self.mode == 'constant':
            tf_config.mode = 'CONSTANT'
        elif self.mode == 'reflect':
            tf_config.mode = 'REFLECT'
        elif self.mode == 'edge':
            tf_config.mode = 'SYMMETRIC'

        if self.data_format == 'NCHW':
            tf_config.input_shape = [
                self.input_shape[0], self.input_shape[2], self.input_shape[3],
                self.input_shape[1]
            ]
        return tf_config


class PDPad2d(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='input',
                shape=config.input_shape,
                dtype=config.input_dtype,
                lod_level=0)
            data.stop_gradient = False
            result = fluid.layers.pad2d(
                input=data,
                paddings=config.paddings,
                mode=config.mode,
                pad_value=config.pad_value,
                data_format=config.data_format)

            self.feed_vars = [data]
            self.fetch_vars = [result]


class TFPad2d(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        data = self.placeholder(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        result = tf.pad(tensor=data,
                        paddings=config.paddings,
                        mode=config.mode,
                        constant_values=config.pad_value)

        self.feed_list = [x]
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(PDPad2d(), TFPad2d(), config=Pad2dConfig())
