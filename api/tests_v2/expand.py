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

from __future__ import division
from common_import import *


class PDExpand(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = paddle.expand(x=x, shape=config.expand_shape)

        self.feed_vars = [x]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TFExpand(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        assert len(config.x_shape) == len(config.expand_shape)
        expand_times = []
        for i in range(len(config.x_shape)):
            if config.x_shape[i] == -1:
                expand_times.append(1)
            else:
                expand_times.append(config.expand_shape[i] // config.x_shape[i])
        result = tf.tile(input=x, multiples=expand_times)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(PDExpand(), TFExpand(), config=APIConfig("expand"))
