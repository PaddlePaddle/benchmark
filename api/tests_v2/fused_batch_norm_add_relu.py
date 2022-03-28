#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from batch_norm import BatchNormConfig


class FusedBatchNormAddReluConfig(BatchNormConfig):
    def __init__(self):
        super(FusedBatchNormAddReluConfig,
              self).__init__("fused_batch_norm_add_relu")
        self.alias_name = "batch_norm"


class PDFusedBatchNormAddRelu(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        def _create_parameter(name, value, stop_gradient):
            param = paddle.create_parameter(
                name=name,
                shape=[config.num_channels],
                dtype=config.x_dtype,
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(value)))
            param.stop_gradient = stop_gradient
            return param

        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.x_shape, dtype=config.x_dtype)

        running_mean = _create_parameter(
            name='running_mean', value=0.5, stop_gradient=True)
        running_var = _create_parameter(
            name='running_var', value=0.1, stop_gradient=True)

        scale = _create_parameter(name='scale', value=0.5, stop_gradient=False)
        bias = _create_parameter(name='bias', value=0.1, stop_gradient=False)

        bn_out = paddle.nn.functional.batch_norm(
            x=x,
            running_mean=running_mean,
            running_var=running_var,
            weight=scale,
            bias=bias,
            epsilon=config.epsilon,
            momentum=config.momentum,
            training=config.training,
            data_format=config.data_format)
        add_out = bn_out + y
        relu_out = paddle.nn.functional.relu(add_out)

        self.feed_vars = [x, y]
        self.fetch_vars = [bn_out, add_out, relu_out]
        if config.backward:
            self.append_gradients(relu_out, [x, scale, bias, bn_out, add_out])


class TFFusedBatchNormAddRelu(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.x_shape, dtype=config.x_dtype)
        bn = tf.keras.layers.BatchNormalization(
            axis=config.axis,
            momentum=config.momentum,
            epsilon=config.epsilon,
            beta_initializer=tf.constant_initializer(0.1),
            gamma_initializer=tf.constant_initializer(0.5),
            moving_mean_initializer=tf.constant_initializer(0.5),
            moving_variance_initializer=tf.constant_initializer(0.1))
        bn_out = bn(x, training=config.training)
        add_out = bn_out + y
        relu_out = tf.nn.relu(add_out)

        self.feed_list = [x, y]
        self.fetch_list = [bn_out, add_out, relu_out]
        if config.backward:
            self.append_gradients(relu_out,
                                  [x, bn.gamma, bn.beta, bn_out, add_out])


if __name__ == '__main__':
    test_main(
        PDFusedBatchNormAddRelu(),
        TFFusedBatchNormAddRelu(),
        config=FusedBatchNormAddReluConfig())
