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


class WhileLoopConfig(APIConfig):
    def __init__(self):
        super(WhileLoopConfig, self).__init__('while_loop')

    def init_from_json(self, filename, config_id=0):
        super(WhileLoopConfig, self).init_from_json(filename, config_id)
        num_flatten_dims = self.num_flatten_dims
        if num_flatten_dims < 0:
            num_flatten_dims = num_flatten_dims + len(self.input_shape)
        row = 1
        col = 1
        for i in range(len(self.input_shape)):
            if i < self.num_flatten_dims:
                row = row * self.input_shape[i]
            else:
                col = col * self.input_shape[i]
        self.input_shape = [row, col]
        self.num_flatten_dims = -1

    def to_tensorflow(self):
        tf_config = self
        if self.act == "relu":
            tf_config.act = tf.nn.relu
        return tf_config


class PDWhileLoop(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):

            def cond(i, loop_len, input, result):
                return i < loop_len

            def body(i, loop_len, input, result):
                result = fluid.layers.fc(
                    input=input,
                    size=config.size,
                    num_flatten_dims=-1,
                    param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.ConstantInitializer(
                            0.5)),
                    bias_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.ConstantInitializer(
                            0.1)),
                    act=config.act)
                fluid.layers.increment(i)
                return [i, loop_len, input, result]

            input = fluid.data(
                name="input",
                shape=config.input_shape,
                dtype=config.input_dtype,
                lod_level=0)
            input.stop_gradient = False
            i = fluid.layers.zeros(shape=[1], dtype='int64')
            loop_len = fluid.layers.ones(shape=[1], dtype='int64')
            result = fluid.layers.zeros(
                shape=[config.input_shape[0], config.size],
                dtype=config.input_dtype)
            _, _, _, results = fluid.layers.while_loop(
                cond, body, [i, loop_len, input, result])
            self.feed_vars = [input]
            self.fetch_vars = [results]
            if config.backward:
                self.append_gradients(results, [input])


class TFWhileLoop(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        def cond(i, loop_len, input, result):
            return i < loop_len

        def body(i, loop_len, input, result):
            if tf.__version__ <= "1.15.0":
                result = tf.contrib.layers.fully_connected(
                    inputs=input,
                    num_outputs=config.size,
                    weights_initializer=tf.constant_initializer(0.5),
                    biases_initializer=tf.constant_initializer(0.1),
                    activation_fn=config.act)
            else:
                result = tf.compat.v1.layers.dense(
                    inputs=input,
                    units=config.size,
                    activation=config.act,
                    use_bias=True,
                    kernel_initializer=tf.constant_initializer(0.5),
                    bias_initializer=tf.constant_initializer(0.1))
            return [i + 1, loop_len, input, result]

        input = self.placeholder(
            name="input", shape=config.input_shape, dtype=config.input_dtype)
        i = tf.constant(0)
        loop_len = tf.constant(1)
        result = tf.zeros(
            shape=[config.input_shape[0], config.size],
            dtype=config.input_dtype)
        if tf.__version__ <= "1.15.0":
            _, _, _, results = tf.while_loop(cond, body,
                                             [i, loop_len, input, result])
        else:
            _, _, _, results = tf.compat.v1.while_loop(
                cond, body, [i, loop_len, input, result])
        self.feed_list = [input]
        self.fetch_list = [results]
        if config.backward:
            self.append_gradients(results, [input])


if __name__ == '__main__':
    test_main(PDWhileLoop(), TFWhileLoop(), config=WhileLoopConfig())
