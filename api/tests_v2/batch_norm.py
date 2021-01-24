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
        # tf's batch_norm does not have data_format param, it only support NHWC format.
        if self.data_format == "NCHW":
            print(
                "Warning:\n"
                "  1. tf's batch_norm does not have data_format param, it only support NHWC format.\n"
            )
            self.run_tf = False

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
            tf_config.axes = [0, 1, 2]
        else:
            tf_config.axes = [0]
        return tf_config


class PDBatchNorm(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)

        running_mean = paddle.create_parameter(
            name='running_mean',
            shape=[config.num_channels],
            dtype=config.x_dtype,
            attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.5)))
        running_mean.stop_gradient = True
        running_var = paddle.create_parameter(
            name='running_var',
            shape=[config.num_channels],
            dtype=config.x_dtype,
            attr=paddle.ParamAttr(
                initializer=paddle.nn.initializer.Constant(0.1)))
        running_var.stop_gradient = True

        scale = self.variable(
            name='scale', shape=[config.num_channels], dtype=config.x_dtype)
        bias = self.variable(
            name='bias', shape=[config.num_channels], dtype=config.x_dtype)

        result = paddle.nn.functional.batch_norm(
            x=x,
            running_mean=running_mean,
            running_var=running_var,
            weight=scale,  # scale
            bias=bias,  # bias
            epsilon=config.epsilon,
            momentum=config.momentum,
            training=config.training,
            data_format=config.data_format)

        self.feed_vars = [x, scale, bias]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x, scale, bias])


class TFBatchNorm(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        scale = self.variable(
            name='scale', shape=[config.num_channels], dtype=config.x_dtype)
        bias = self.variable(
            name='bias', shape=[config.num_channels], dtype=config.x_dtype)
        mean, var = tf.nn.moments(
            x=x, axes=config.axes, shift=None, keepdims=False)
        result = tf.nn.batch_normalization(
            x=x,
            mean=mean,
            variance=var,
            offset=bias,
            scale=scale,
            variance_epsilon=config.epsilon)

        self.feed_list = [x, scale, bias]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, scale, bias])


if __name__ == '__main__':
    test_main(PDBatchNorm(), TFBatchNorm(), config=BatchNormConfig())
