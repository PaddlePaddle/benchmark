#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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


class FCConfig(APIConfig):
    def __init__(self):
        super(FCConfig, self).__init__('fc')

    def init_from_json(self, filename, config_id=0):
        super(FCConfig, self).init_from_json(filename, config_id)
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


class PDFC(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        input = self.variable(
            name="input", shape=config.input_shape, dtype=config.input_dtype)
        result = fluid.layers.fc(
            input=input,
            size=config.size,
            num_flatten_dims=-1,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.ConstantInitializer(0.5)),
            bias_attr=fluid.ParamAttr(
                initializer=fluid.initializer.ConstantInitializer(0.1)),
            act=config.act)

        self.feed_vars = [input]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [input])


class TFFC(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name="input",
            shape=config.input_shape,
            dtype=config.input_dtype,
            value=config.input_data)
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

        self.feed_list = [input]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    test_main(PDFC(), TFFC(), config=FCConfig())
