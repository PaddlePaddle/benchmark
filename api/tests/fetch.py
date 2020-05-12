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


class PDFetch(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.layers.create_parameter(
                shape=config.x_shape,
                dtype=config.x_dtype,
                attr=fluid.ParamAttr(
                    initializer=fluid.initializer.NumpyArrayInitializer(
                        config.x_data)))

            self.feed_vars = []
            self.fetch_vars = [data]


class TFFetch(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        data = tf.Variable(config.x_data)

        self.feed_list = []
        self.fetch_list = [data]


if __name__ == '__main__':
    test_main(PDFetch(), TFFetch(), config=APIConfig("fetch"))
