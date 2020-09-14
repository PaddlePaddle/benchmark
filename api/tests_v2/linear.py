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


class LinearConfig(APIConfig):
    def __init__(self):
        super(LinearConfig, self).__init__('linear')

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(LinearConfig, self).init_from_json(filename, config_id,
                                                 unknown_dim)
        self.feed_spec = [
            {
                "range": [0, 10]
            },  # x
            {
                "range": [0.5, 0.5]
            },  # weight
            {
                "range": [0.1, 0.1]
            }  # bias
        ]

    def to_tensorflow(self):
        tf_config = super(LinearConfig, self).to_tensorflow()
        tf_config.size = self.weight_shape[-1]
        return tf_config


class PDLinear(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name="x", shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name="weight",
            shape=config.weight_shape,
            dtype=config.weight_dtype)
        bias = self.variable(
            name="bias", shape=config.bias_shape, dtype=config.bias_dtype)
        result = paddle.nn.functional.linear(x=x, weight=weight, bias=bias)

        self.feed_vars = [x, weight, bias]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x, weight, bias])


class TFLinear(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name="x", shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name="weight",
            shape=config.weight_shape,
            dtype=config.weight_dtype)
        bias = self.variable(
            name="bias", shape=config.bias_shape, dtype=config.bias_dtype)
        result = tf.compat.v1.layers.dense(
            inputs=x,
            units=config.size,
            activation=None,
            use_bias=True,
            kernel_initializer=tf.constant_initializer(0.5),
            bias_initializer=tf.constant_initializer(0.1))

        self.feed_list = [x, weight, bias]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, weight, bias])


if __name__ == '__main__':
    test_main(PDLinear(), TFLinear(), config=LinearConfig())
