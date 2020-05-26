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

    def init_from_json(self, filename, config_id=0):
        super(BatchNormConfig, self).init_from_json(filename, config_id)
        if self.input_shape[0] == -1:
            self.input_shape[0] = 16
        # TFBatchNorm does not have data_layout param, it only support NHWC format.
        if self.data_layout == "NCHW":
            self.run_tf = False
        if len(self.input_shape) == 4:
            if self.data_layout == "NCHW":
                self.num_channels = self.input_shape[1]
            else:
                self.num_channels = self.input_shape[3]
        else:
            self.num_channels = self.input_shape[1]

    def to_tensorflow(self):
        tf_config = self
        if len(tf_config.input_shape) == 4:
            tf_config.axes = [0, 1, 2]
        else:
            tf_config.axes = [0]
        return tf_config


class PDBatchNorm(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            input = fluid.data(
                name='input',
                shape=config.input_shape,
                dtype=config.input_dtype,
                lod_level=0)
            scale = fluid.layers.create_parameter(
                name='scale',
                shape=[config.num_channels],
                dtype=config.input_dtype)
            bias = fluid.layers.create_parameter(
                name='bias',
                shape=[config.num_channels],
                dtype=config.input_dtype)
            input.stop_gradient = False
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
        input = tf.placeholder(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        scale = tf.placeholder(
            name='scale',
            shape=[config.num_channels],
            dtype=config.input_dtype)
        bias = tf.placeholder(
            name='bias', shape=[config.num_channels], dtype=config.input_dtype)
        mean, var = tf.nn.moments(
            x=input, axes=config.axes, shift=None, keepdims=False)
        result = tf.nn.batch_normalization(
            x=input,
            mean=mean,
            variance=var,
            offset=bias,
            scale=scale,
            variance_epsilon=config.epsilon)

        self.feed_list = [input, scale, bias]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input, scale, bias])


def register_api():
    REGISTER_API_INFO['batch_norm'] = ['batch_norm', 'batch_norm.json']


if __name__ == '__main__':
    test_main(PDBatchNorm(), TFBatchNorm(), config=BatchNormConfig())
