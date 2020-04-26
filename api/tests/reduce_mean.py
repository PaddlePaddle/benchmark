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


class PDReduceMean(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            input = fluid.data(
                name='input',
                shape=config.input_shape,
                dtype=config.input_dtype,
                lod_level=0)
            input.stop_gradient = False
            result = fluid.layers.reduce_mean(
                input=input, dim=config.dim, keep_dim=config.keep_dim)

            self.feed_vars = [input]
            self.fetch_vars = [result]
            if config.backward:
                self.append_gradients(result, [input])


class TFReduceMean(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.placeholder(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        result = tf.reduce_mean(
            input_tensor=input, axis=config.dim, keepdims=config.keep_dim)

        self.feed_list = [input]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    test_main(PDReduceMean(), TFReduceMean(), config=APIConfig("reduce_mean"))
