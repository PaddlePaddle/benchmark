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
        super(ArgsortConfig, self).__init__('argsort')

    def to_tensorflow(self):
        if self.descending == True:
            setattr(self, direction, 'DESCENDING')
        else:
            setattr(self, direction, 'ASCENDING')
        return self


class PDArgsort(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data',
                shape=config.input_shape,
                dtype=config.input_dtype,
                lod_level=0)
            data.stop_gradient = False
            result = fluid.layers.argsort(
                input=data, axis=config.axis, descending=config.descending)

            self.feed_vars = [data]
            self.fetch_vars = [result]
            if config.backward:
                self.append_gradients(result, [data])


class TFArgsort(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        data = self.placeholder(
            name='data', shape=config.input_shape, dtype=config.input_dtype)
        result = tf.argsort(
            values=data, axis=config.axis, direction=config.direction)

        self.feed_list = [data]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [data])


if __name__ == '__main__':
    test_main(PDArgsort(), TFArgsort(), config=ArgsortConfig())
