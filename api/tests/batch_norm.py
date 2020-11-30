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
        if self.data_layout == "NCHW" and len(self.input_shape) == 4:
            self.feed_spec = [
                {
                    "range": [-1, 1],
                    "permute": [0, 2, 3, 1]
                },  # input
                {
                    "range": [-1, 1],
                },  # scale
                {
                    "range": [-1, 1],
                }  # bias
            ]

        if len(self.input_shape) == 4:
            if self.data_layout == "NCHW":
                self.num_channels = self.input_shape[1]
            else:
                self.num_channels = self.input_shape[3]
        else:
            self.num_channels = self.input_shape[1]

    def to_tensorflow(self):
        tf_config = super(BatchNormConfig, self).to_tensorflow()
        if self.data_layout == "NCHW" and len(self.input_shape) == 4:
            print(
                "Warning:\n"
                "  1. tf's batch_norm does not have data_format param, it only support NHWC format. The benchmark test is actually running with NHWC format.\n"
            )
            tf_config.data_layout = "NHWC"
            tf_config.input_shape = [
                self.input_shape[0], self.input_shape[2], self.input_shape[3],
                self.input_shape[1]
            ]

        if len(tf_config.input_shape) == 4:
            tf_config.axes = [0, 1, 2]
        else:
            tf_config.axes = [0]
        return tf_config


class PDBatchNorm(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        input = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        scale = self.variable(
            name='scale',
            shape=[config.num_channels],
            dtype=config.input_dtype)
        bias = self.variable(
            name='bias', shape=[config.num_channels], dtype=config.input_dtype)
        result = fluid.layers.batch_norm(
            input=input,
            act=None,
            is_test=False,
            momentum=0.9,
            epsilon=config.epsilon,
            param_attr="scale",
            bias_attr="bias",
            data_layout=config.data_layout)

        self.feed_vars = [input, scale, bias]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [input, scale, bias])


class TFBatchNorm(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        bias = self.variable(
            name='bias', shape=[config.num_channels], dtype=config.input_dtype)
        scale = self.variable(
            name='scale',
            shape=[config.num_channels],
            dtype=config.input_dtype)
        mean, var = tf.nn.moments(
            x=input, axes=config.axes, shift=None, keepdims=False)
        result = tf.nn.batch_normalization(
            x=input,
            mean=mean,
            variance=var,
            offset=bias,  # beta
            scale=scale,  # gamma
            variance_epsilon=config.epsilon)

        self.feed_list = [input, scale, bias]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input, scale, bias])


if __name__ == '__main__':
    test_main(PDBatchNorm(), TFBatchNorm(), config=BatchNormConfig())
