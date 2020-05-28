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


class PDCond(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        def true_fn():
            return fluid.layers.elementwise_mul(x, y)

        def false_fn():
            return fluid.layers.elementwise_div(x, y)

        ten_var = fluid.layers.fill_constant(
            shape=config.input_shape, dtype=config.input_dtype, value=10)

        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)
        input = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        pred = fluid.layers.less_than(input, ten_var)
        result = fluid.layers.cond(pred, true_fn, false_fn)

        self.feed_vars = [x, y, input]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x, y, input])


class TFCond(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        def true_fn():
            return tf.multiply(x, y)

        def false_fn():
            return tf.divide(x, y)

        ten_var = tf.constant(
            10, shape=config.input_shape, dtype=config.input_dtype)

        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)
        input = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        pred = tf.less(input, ten_var)
        result = tf.cond(tf.reshape(pred, []), true_fn, false_fn)

        self.feed_list = [x, y, input]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, y, input])


if __name__ == '__main__':
    test_main(PDCond(), TFCond(), config=APIConfig('cond'))
