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


class PDFillConstant(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        result = fluid.layers.fill_constant(
            shape=config.shape, dtype=config.dtype, value=config.value)

        self.feed_vars = []
        self.fetch_vars = [result]


class TFFillConstant(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        result = tf.constant(
            shape=config.shape,
            dtype=tf.as_dtype(config.dtype),
            value=config.value)

        self.feed_list = []
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(
        PDFillConstant(), TFFillConstant(), config=APIConfig("fill_constant"))
