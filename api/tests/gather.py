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


class PDGather(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            input = fluid.data(
                name='input',
                shape=config.input_shape,
                dtype=config.input_dtype,
                lod_level=0)
            index = fluid.data(
                name='index',
                shape=config.index_shape,
                dtype=config.index_dtype,
                lod_level=0)
            input.stop_gradient = False
            index.stop_gradient = False
            result = fluid.layers.gather(
                input=input, index=index, overwrite=config.overwrite)

            self.feed_vars = [input, index]
            self.fetch_vars = [result]
            if config.backward:
                self.append_gradients(result, [input, index])


class TFGather(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        params = self.placeholder(
            name='params', shape=config.input_shape, dtype=config.input_dtype)
        indices = self.placeholder(
            name='indices', shape=config.index_shape, dtype=config.index_dtype)
        result = tf.gather(params=params, indices=indices)

        self.feed_list = [params, indices]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [params, indices])


if __name__ == '__main__':
    test_main(PDGather(), TFGather(), config=APIConfig("gather"))
