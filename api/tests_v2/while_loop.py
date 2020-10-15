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
try:
    tf.compat.v1.disable_v2_behavior()
except Exception:
    pass


class WhileLoopConfig(APIConfig):
    def __init__(self):
        super(WhileLoopConfig, self).__init__('while_loop')
        self.alias_config = APIConfig("linear")

    def to_tensorflow(self):
        tf_config = super(WhileLoopConfig, self).to_tensorflow()
        tf_config.size = self.alias.weight_shape[-1]
        return tf_config


class PDWhileLoop(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        def cond(i, loop_len, x, result):
            return i < loop_len

        def body(i, loop_len, x, result):
            result = paddle.nn.functional.linear(x=x, weight=weight, bias=bias)
            paddle.increment(i)
            return [i, loop_len, x, result]

        x = self.variable(
            name="x", shape=config.alias.x_shape, dtype=config.alias.x_dtype)
        weight = paddle.create_parameter(
            shape=config.alias.weight_shape,
            dtype=config.alias.weight_dtype,
            name="weight",
            attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.5)))
        bias = paddle.create_parameter(
            shape=config.alias.bias_shape,
            dtype=config.alias.bias_dtype,
            name="bias",
            attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.1)))
        i = paddle.zeros(shape=[1], dtype='int64')
        loop_len = paddle.ones(shape=[1], dtype='int64')
        result = paddle.zeros(
            shape=config.alias.x_shape[:-1] + config.alias.weight_shape[-1:],
            dtype=config.alias.x_dtype)
        result.stop_gradient = False
        _, _, _, results = paddle.static.nn.while_loop(
            cond, body, [i, loop_len, x, result])
        self.feed_vars = [x]
        self.fetch_vars = [results]
        if config.backward:
            self.append_gradients(results, [x])


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
                    activation_fn=None)
            else:
                result = tf.compat.v1.layers.dense(
                    inputs=input,
                    units=config.size,
                    activation=None,
                    use_bias=True,
                    kernel_initializer=tf.constant_initializer(0.5),
                    bias_initializer=tf.constant_initializer(0.1))
            return [i + 1, loop_len, input, result]

        input = self.variable(
            name="input",
            shape=config.alias.x_shape,
            dtype=config.alias.x_dtype)
        i = tf.constant(0)
        loop_len = tf.constant(1)
        result = tf.zeros(
            shape=config.alias.x_shape[:-1] + config.alias.weight_shape[-1:],
            dtype=config.alias.x_dtype)
        _, _, _, results = tf.while_loop(cond, body,
                                         [i, loop_len, input, result])
        self.feed_list = [input]
        self.fetch_list = [results]
        if config.backward:
            self.append_gradients(results, [input])


if __name__ == '__main__':
    test_main(PDWhileLoop(), TFWhileLoop(), config=WhileLoopConfig())
