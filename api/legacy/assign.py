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

import numpy as np
from common_import import *


class PDAssign(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        data = self.variable(
            name='data', shape=config.input_shape, dtype=config.input_dtype)
        result = fluid.layers.assign(input=data)

        self.feed_vars = [data]
        self.fetch_vars = [result]


class TFAssign(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        data = self.variable(
            name='data', shape=config.input_shape, dtype=config.input_dtype)
        ref = self.variable(
            name='target',
            shape=config.input_shape,
            dtype=config.input_dtype,
            value=np.zeros(config.input_shape).astype(config.input_dtype))
        result = ref.assign(data)

        self.feed_list = [data]
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(PDAssign(), TFAssign(), config=APIConfig("assign"))
