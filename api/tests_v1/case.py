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


class PDCase(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        five_var = fluid.layers.fill_constant(
            shape=config.input_shape, dtype=config.input_dtype, value=5)
        ten_var = fluid.layers.fill_constant(
            shape=config.input_shape, dtype=config.input_dtype, value=10)
        one_var = fluid.layers.ones(
            shape=config.input_shape, dtype=config.input_dtype)

        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)

        def f1():
            return fluid.layers.elementwise_add(x=x, y=y)

        def f2():
            return fluid.layers.elementwise_sub(x=x, y=y)

        def f3():
            return fluid.layers.elementwise_mul(x=x, y=y)

        pred_1 = fluid.layers.less_than(one_var, five_var)
        pred_2 = fluid.layers.greater_than(one_var, ten_var)

        result = fluid.layers.case(
            pred_fn_pairs=[(pred_1, f1), (pred_2, f2)], default=f3)
        self.feed_vars = [x, y]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x, y])


class TFCase(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        five_var = tf.constant(
            5, shape=config.input_shape, dtype=config.input_dtype)
        ten_var = tf.constant(
            10, shape=config.input_shape, dtype=config.input_dtype)
        one_var = tf.ones(shape=config.input_shape, dtype=config.input_dtype)

        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)

        def f1():
            return tf.add(x, y)

        def f2():
            return tf.subtract(x, y)

        def f3():
            return tf.multiply(x, y)

        pred_1 = tf.less(one_var, five_var)
        pred_2 = tf.greater(one_var, ten_var)

        result = tf.case(
            [(tf.reshape(pred_1, []), f1), (tf.reshape(pred_2, []), f2)],
            default=f3)
        self.feed_list = [x, y]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, y])


if __name__ == '__main__':
    test_main(PDCase(), TFCase(), config=APIConfig('case'))
