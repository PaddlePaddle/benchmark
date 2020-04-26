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


class PDSigmoid(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data',
                shape=config.x_shape,
                dtype=config.x_dtype,
                lod_level=0)
            data.stop_gradient = False
            result = fluid.layers.sigmoid(x=data)

            self.feed_vars = [data]
            self.fetch_vars = [result]
            if config.backward:
                self.append_gradients(result, [data])


class TFSigmoid(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        data = self.placeholder(
            name='data', shape=config.x_shape, dtype=config.x_dtype)
        result = tf.sigmoid(x=data)

        self.feed_list = [data]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [data])


if __name__ == '__main__':
    test_main(PDSigmoid(), TFSigmoid(), config=APIConfig("sigmoid"))
