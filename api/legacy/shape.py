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


class PDShape(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        data = self.variable(
            name='data', shape=config.input_shape, dtype=config.input_dtype)
        result = fluid.layers.shape(input=data)

        self.feed_vars = [data]
        self.fetch_vars = [result]


class TFShape(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        data = self.variable(
            name='data', shape=config.input_shape, dtype=config.input_dtype)
        result = tf.shape(input=data)

        self.feed_list = [data]
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(PDShape(), TFShape(), config=APIConfig("shape"))
