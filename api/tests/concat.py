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


class PDConcat(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            inputs = []
            for i in range(len(config.input_shape)):
                input_i = fluid.data(
                    name='input_' + str(i),
                    shape=config.input_shape[i],
                    dtype=config.input_dtype[i],
                    lod_level=0)
                input_i.stop_gradient = False
                inputs.append(input_i)
            result = fluid.layers.concat(input=inputs, axis=config.axis)

            self.feed_vars = inputs
            self.fetch_vars = [result]
            if config.backward:
                self.append_gradients(result, inputs)


class TFConcat(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        inputs = []
        for i in range(len(config.input_shape)):
            input_i = self.placeholder(
                name='input_' + str(i),
                shape=config.input_shape[i],
                dtype=config.input_dtype[i])
            inputs.append(input_i)
        result = tf.concat(values=inputs, axis=config.axis)

        self.feed_list = inputs
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, inputs)


def register_api():
    REGISTER_API_INFO['concat'] = ['concat', 'concat.json']


if __name__ == '__main__':
    test_main(PDConcat(), TFConcat(), config=APIConfig("concat"))
