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
from fc import FCConfig


class WhileLoopConfig(APIConfig):
    def __init__(self):
        super(WhileLoopConfig, self).__init__('while_loop')
        self.alias_config = FCConfig()


class PDWhileLoop(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):

            def cond(i, loop_len, input, result):
                return i < loop_len

            def body(i, loop_len, input, result):
                result = fluid.layers.fc(
                    input=input,
                    size=config.alias_config.size,
                    num_flatten_dims=-1,
                    param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.ConstantInitializer(
                            0.5)),
                    bias_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.ConstantInitializer(
                            0.1)),
                    act=config.alias_config.act)
                fluid.layers.increment(i)
                return [i, loop_len, input, result]

            input = fluid.data(
                name="input",
                shape=config.alias_config.input_shape,
                dtype=config.alias_config.input_dtype,
                lod_level=0)
            input.stop_gradient = False
            i = fluid.layers.zeros(shape=[1], dtype='int64')
            loop_len = fluid.layers.ones(shape=[1], dtype='int64')
            result = fluid.layers.zeros(
                shape=[
                    config.alias_config.input_shape[0],
                    config.alias_config.size
                ],
                dtype=config.alias_config.input_dtype)
            _, _, _, results = fluid.layers.while_loop(
                cond, body, [i, loop_len, input, result])
            self.feed_vars = [input]
            self.fetch_vars = [results]
            if config.alias_config.backward:
                self.append_gradients(results, [input])


class TFWhileLoop(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        def cond(i, loop_len, input, result):
            return i < loop_len

        def body(i, loop_len, input, result):
            if tf.__version__ <= "1.15.0":
                result = tf.contrib.layers.fully_connected(
                    inputs=input,
                    num_outputs=config.alias_config.size,
                    weights_initializer=tf.constant_initializer(0.5),
                    biases_initializer=tf.constant_initializer(0.1),
                    activation_fn=config.alias_config.act)
            else:
                result = tf.compat.v1.layers.dense(
                    inputs=input,
                    units=config.alias_config.size,
                    activation=config.alias_config.act,
                    use_bias=True,
                    kernel_initializer=tf.constant_initializer(0.5),
                    bias_initializer=tf.constant_initializer(0.1))
            return [i + 1, loop_len, input, result]

        input = self.placeholder(
            name="input",
            shape=config.alias_config.input_shape,
            dtype=config.alias_config.input_dtype)
        i = tf.constant(0)
        loop_len = tf.constant(1)
        result = tf.zeros(
            shape=[
                config.alias_config.input_shape[0], config.alias_config.size
            ],
            dtype=config.alias_config.input_dtype)
        if tf.__version__ <= "1.15.0":
            _, _, _, results = tf.while_loop(cond, body,
                                             [i, loop_len, input, result])
        else:
            _, _, _, results = tf.compat.v1.while_loop(
                cond, body, [i, loop_len, input, result])
        self.feed_list = [input]
        self.fetch_list = [results]
        if config.alias_config.backward:
            self.append_gradients(results, [input])


def register_api():
    REGISTER_API_INFO['while_loop'] = ['while_loop', 'fc.json']


if __name__ == '__main__':
    test_main(PDWhileLoop(), TFWhileLoop(), config=WhileLoopConfig())
