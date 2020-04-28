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


class ArgsortConfig(APIConfig):
    def __init__(self):
        super(ArgsortConfig, self).__init__("argsort")

    def to_tensorflow(self):
        tf_config = self
        if self.descending:
            tf_config.direction = "DESCENDING"
        else:
            tf_config.direction = "ASCENDING"
        return tf_config


class PDArgsort(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            input = fluid.data(
                name='input',
                shape=config.input_shape,
                dtype=config.input_dtype,
                lod_level=0)
            input.stop_gradient = False
            result, indices = fluid.layers.argsort(
                input=input, axis=config.axis, descending=config.descending)

            self.feed_vars = [input]
            self.fetch_vars = [indices]


class TFArgsort(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.placeholder(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        indices = tf.argsort(
            values=input,
            axis=config.axis,
            direction=config.direction,
            stable=False)

        self.feed_list = [input]
        self.fetch_list = [indices]


if __name__ == '__main__':
    test_main(PDArgsort(), TFArgsort(), config=ArgsortConfig())
