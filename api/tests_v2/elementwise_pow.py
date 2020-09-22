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


class PDElementwisePow(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)
        result = paddle.elementwise_pow(x=x, y=y, axis=config.axis, act=None)

        self.feed_vars = [x, y]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x, y])


class TFElementwisePow(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)
        if hasattr(config, "x_shape_unsqueezed"):
            x_reshape = tf.reshape(tensor=x, shape=config.x_shape_unsqueezed)
        else:
            x_reshape = x
        if hasattr(config, "y_shape_unsqueezed"):
            y_reshape = tf.reshape(tensor=y, shape=config.y_shape_unsqueezed)
        else:
            y_reshape = y
        result = tf.math.pow(x=x_reshape, y=y_reshape)

        self.feed_list = [x, y]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, y])


if __name__ == '__main__':
    test_main(
        PDElementwisePow(),
        TFElementwisePow(),
        config=APIConfig("elementwise_pow"))
