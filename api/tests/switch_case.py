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


class PDSwitchCase(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            x = fluid.data(
                name='x', shape=config.x_shape, dtype=config.x_dtype)
            y = fluid.data(
                name='y', shape=config.y_shape, dtype=config.y_dtype)
            input = fluid.data(
                name='input',
                shape=config.input_shape,
                dtype=config.input_dtype)
            x.stop_gradient = False
            y.stop_gradient = False
            input.stop_gradient = False

            def f1():
                return fluid.layers.elementwise_add(x=x, y=y)

            def f2():
                return fluid.layers.elementwise_sub(x=x, y=y)

            def f3():
                return fluid.layers.elementwise_mul(x=x, y=y)

            result = fluid.layers.switch_case(
                branch_index=input, branch_fns={0: f1,
                                                1: f2}, default=f3)
            self.feed_vars = [x, y, input]
            self.fetch_vars = [result]
            if config.backward:
                self.append_gradients(result, [x, y, input])


class TFSwitchCase(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.placeholder(
            name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.placeholder(
            name='y', shape=config.y_shape, dtype=config.y_dtype)
        input = self.placeholder(
            name='input', shape=config.input_shape, dtype=config.input_dtype)

        def f1():
            return tf.add(x, y)

        def f2():
            return tf.subtract(x, y)

        def f3():
            return tf.multiply(x, y)

        result = tf.switch_case(
            branch_index=tf.reshape(input, []),
            branch_fns={0: f1,
                        1: f2},
            default=f3)
        self.feed_list = [x, y, input]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, y, input])


if __name__ == '__main__':
    test_main(PDSwitchCase(), TFSwitchCase(), config=APIConfig('switch_case'))
