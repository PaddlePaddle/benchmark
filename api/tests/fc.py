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

import paddle.fluid as fluid
from main import test_main

import sys
sys.path.append("..")
from common import paddle_api_benchmark as paddle_api
from common import tensorflow_api_benchmark as tensorflow_api
from common import api_param


class FCConfig(api_param.APIConfig):
    def __init__(self, input_shape=[1,256,6,6], size=1, num_flatten_dims=-1, act=None, dtype='float32'):
        super(FCConfig, self).__init__('fc','')
        row = 1
        col = 1
        if num_flatten_dims < 0:
            num_flatten_dims = num_flatten_dims + len(input_shape)
        for i in range(len(input_shape)):
            if i < num_flatten_dims:
                row = row * input_shape[i]
            else:
                col = col * input_shape[i]
        self.input_shape = [row, col]
        self.size = size
        self.num_flatten_dims = num_flatten_dims
        self.act=act
        self.input_dtype=dtype

class PDFC(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False, dtype=None):
        import paddle.fluid as fluid

        self.name = "fc"
        self.main_program=fluid.Program()
        self.startup_program=fluid.Program()
        with fluid.program_guard(self.main_program, self.startup_program):
            input = fluid.data(
                name=FCconfig.input_name, shape=FCconfig.input_shape, dtype=FCconfig.input_dtype, lod_level=0)
            print(FCconfig.input_name)
            print(FCconfig.input_shape)
            input.stop_gradient = False
            result = fluid.layers.fc(
                input=input,
                size=FCconfig.size,
                num_flatten_dims=FCconfig.num_flatten_dims,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.ConstantInitializer(0.5)),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.ConstantInitializer(0.1)),
                act=FCconfig.act)

            self.feed_vars = [input]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [input])


class TFFC(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False, dtype=None):
        import tensorflow as tf

        self.name = "fc"
        self.allow_growth = True

        input = tf.placeholder(name=FCconfig.input_name, shape=FCconfig.input_shape, dtype=tf.as_dtype(FCconfig.input_dtype))
        result = tf.contrib.layers.fully_connected(
            inputs=input,
            num_outputs=FCconfig.size,
            weights_initializer=tf.constant_initializer(0.5),
            biases_initializer=tf.constant_initializer(0.1),
            activation_fn=FCconfig.act)

        self.feed_list = [input]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    FCconfig = FCConfig()
    test_main(PDFC(), TFFC(), FCconfig, feed_spec=None)
