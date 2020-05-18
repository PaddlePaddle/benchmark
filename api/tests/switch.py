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


class PDSwitch(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            result = fluid.layers.create_global_var(
                shape=config.x_shape,
                value=1,
                dtype=config.x_dtype,
                persistable=True)
            zero_var = fluid.layers.fill_constant(
                shape=config.x_shape, value=0, dtype=config.x_dtype)
            five_var = fluid.layers.fill_constant(
                shape=config.x_shape, value=5, dtype=config.x_dtype)
            ten_var = fluid.layers.fill_constant(
                shape=config.x_shape, value=10, dtype=config.x_dtype)
            input = fluid.data(
                name='input', shape=config.x_shape, dtype=config.x_dtype)
            input.stop_gradient = False

            with fluid.layers.Switch() as switch:
                with switch.case(fluid.layers.less_than(input, zero_var)):
                    fluid.layers.assign(zero_var, result)
                with switch.case(fluid.layers.greater_than(input, five_var)):
                    fluid.layers.assign(five_var, result)
                with switch.default():
                    fluid.layers.assign(ten_var, result)
            self.feed_vars = [input]
            self.fetch_vars = [result]
            if config.backward:
                self.append_gradients(result, [input])


class TFCase(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        data = tf.Variable(tf.ones(shape=config.x_shape, dtype=config.x_dtype))
        zero_var = tf.constant(0, shape=config.x_shape, dtype=config.x_dtype)
        five_var = tf.constant(5, shape=config.x_shape, dtype=config.x_dtype)
        ten_var = tf.constant(10, shape=config.x_shape, dtype=config.x_dtype)
        input = self.placeholder(
            name='input', shape=config.x_shape, dtype=config.x_dtype)

        def f1():
            return tf.compat.v1.assign(ref=data, value=zero_var)

        def f2():
            return tf.compat.v1.assign(ref=data, value=five_var)

        def f3():
            return tf.compat.v1.assign(ref=data, value=ten_var)

        result = tf.case(
            [(tf.reshape(tf.less(input, 0), []), f1),
             (tf.reshape(tf.greater(input, 5), []), f2)],
            default=f3)
        self.feed_list = [input]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    test_main(PDSwitch(), TFCase(), config=APIConfig('switch'))
