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


class BatchNormConfig(APIConfig):
    def __init__(self):
        super(BatchNormConfig, self).__init__('batch_norm')

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(BatchNormConfig, self).init_from_json(filename, config_id,
                                                    unknown_dim)
        if len(self.x_shape) == 4:
            if self.data_format == "NCHW":
                self.num_channels = self.x_shape[1]
            else:
                self.num_channels = self.x_shape[3]
        else:
            self.num_channels = self.x_shape[1]

    def to_tensorflow(self):
        tf_config = super(BatchNormConfig, self).to_tensorflow()
        if len(tf_config.x_shape) == 4:
            tf_config.axis = 1 if self.data_format == "NCHW" else 3
        else:
            tf_config.axis = 1
        return tf_config


class PDBatchNorm(PaddleAPIBenchmarkBase):
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

        running_mean = _create_parameter(
            name='running_mean', value=0.5, stop_gradient=True)
        running_var = _create_parameter(
            name='running_var', value=0.1, stop_gradient=True)

        scale = _create_parameter(name='scale', value=0.5, stop_gradient=False)
        bias = _create_parameter(name='bias', value=0.1, stop_gradient=False)

        result = paddle.nn.functional.batch_norm(
            x=x,
            running_mean=running_mean,
            running_var=running_var,
            weight=scale,
            bias=bias,
            epsilon=config.epsilon,
            momentum=config.momentum,
            training=config.training,
            data_format=config.data_format)

        self.feed_vars = [x]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x, scale, bias])


class TFBatchNorm(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        bn = tf.keras.layers.BatchNormalization(
            axis=config.axis,
            momentum=config.momentum,
            epsilon=config.epsilon,
            beta_initializer=tf.constant_initializer(0.1),
            gamma_initializer=tf.constant_initializer(0.5),
            moving_mean_initializer=tf.constant_initializer(0.5),
            moving_variance_initializer=tf.constant_initializer(0.1))
        result = bn(x, training=config.training)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, bn.gamma, bn.beta])


if __name__ == '__main__':
    test_main(PDBatchNorm(), TFBatchNorm(), config=BatchNormConfig())
