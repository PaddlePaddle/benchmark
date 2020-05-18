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
        with fluid.program_guard(self.main_program, self.startup_program):

            def true_fn():
                return fluid.layers.fill_constant(
                    value=1, shape=config.shape, dtype=config.dtype)

            def false_fn():
                return fluid.layers.fill_constant(
                    value=0, shape=config.shape, dtype=config.dtype)

            ten_var = fluid.layers.fill_constant(
                shape=config.x_shape, value=10, dtype=config.x_dtype)
            input = fluid.data(
                name='input', shape=config.x_shape, dtype=config.x_dtype)
            input.stop_gradient = False

            pred = fluid.layers.less_than(input, ten_var)

            result = fluid.layers.cond(pred, true_fn, false_fn)

            self.feed_vars = [input]
            self.fetch_vars = [result]
            if config.backward:
                self.append_gradients(result, [input])


class TFCond(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        def true_fn():
            return tf.constant(1, shape=config.shape, dtype=config.dtype)

        def false_fn():
            return tf.constant(0, shape=config.shape, dtype=config.dtype)

        ten_var = tf.constant(10, shape=config.x_shape, dtype=config.x_dtype)
        input = self.placeholder(
            name='input', shape=config.x_shape, dtype=config.x_dtype)

        result = tf.cond(
            tf.reshape(tf.less(input, ten_var), []), true_fn, false_fn)

        self.feed_list = [input]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    test_main(PDCond(), TFCond(), config=APIConfig('cond'))
