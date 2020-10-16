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
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)
        zero = paddle.zeros(shape=config.input_shape, dtype=config.input_dtype)

        def f1():
            return paddle.add(x=x, y=y)

        def f2():
            return paddle.subtract(x=x, y=y)

        def f3():
            return paddle.multiply(x=x, y=y)

        result = paddle.static.nn.switch_case(
            branch_index=zero, branch_fns={0: f1,
                                           1: f2}, default=f3)
        self.feed_vars = [x, y]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x, y])


class TFSwitchCase(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)
        zero = tf.zeros(shape=config.input_shape, dtype=config.input_dtype)

        def f1():
            return tf.add(x, y)

        def f2():
            return tf.subtract(x, y)

        def f3():
            return tf.multiply(x, y)

        result = tf.switch_case(
            branch_index=tf.reshape(zero, []),
            branch_fns={0: f1,
                        1: f2},
            default=f3)
        self.feed_list = [x, y]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, y])


if __name__ == '__main__':
    test_main(PDSwitchCase(), TFSwitchCase(), config=APIConfig('switch_case'))
