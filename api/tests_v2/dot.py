#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

class DotConfig(APIConfig):
    def __init__(self):
        super(DotConfig, self).__init__('dot')
        self.run_tf = False

class PaddleDot(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='X', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='Y', shape=config.y_shape, dtype=config.y_dtype)
        result = paddle.dot(x, y)

        self.feed_vars = [x, y]
        self.fetch_vars = [result]

        if config.backward:
            self.append_gradients(result, [x, y])

class TFDot(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='X', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='Y', shape=config.y_shape, dtype=config.y_dtype)
        if len(config.x_shape) == 1:
            axes = [[0], [0]]
        else:
            axes = [[1], [1]] 
        result = tf.tensordot(x, y, axes=axes)

        self.feed_list = [x]
        self.fetch_list = [result]

if __name__ == '__main__':
    test_main(PaddleDot(), TFDot(), config=DotConfig())