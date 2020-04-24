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

from main import test_main

import sys
sys.path.append("..")
from common import paddle_api_benchmark as paddle_api
from common import tensorflow_api_benchmark as tensorflow_api
from common import api_param


class FCConfig(api_param.APIConfig):
    def __init__(self):
        super(FCConfig, self).__init__('fc','')
       # row = 1	        
       # col = 1	
       # if num_flatten_dims < 0:	
       #     num_flatten_dims = num_flatten_dims + len(input_shape)	
       # for i in range(len(input_shape)):	
       #     if i < num_flatten_dims:	
       #         row = row * input_shape[i]	
       #     else:	
       #         col = col * input_shape[i]	
       # self.input_shape = [row, col]	

class PDFC(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, config):
        import paddle.fluid as fluid

        self.name = "fc"
        with fluid.program_guard(self.main_program, self.startup_program):
            input = fluid.data(
                name=config.input_name, shape=config.input_shape, dtype=config.input_dtype, lod_level=0)
            input.stop_gradient = False
            result = fluid.layers.fc(
                input=input,
                size=config.size,
                num_flatten_dims=config.num_flatten_dims,
                param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.ConstantInitializer(0.5)),
                bias_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.ConstantInitializer(0.1)),
                act=config.act)

            self.feed_vars = [input]
            self.fetch_vars = [result]
            if config.backward:
                self.append_gradients(result, [input])


class TFFC(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        import tensorflow as tf

        self.name = "fc"
        self.allow_growth = True

        input = tf.placeholder(name=config.input_name, shape=config.input_shape, dtype=tf.as_dtype(config.input_dtype))
        result = tf.contrib.layers.fully_connected(
            inputs=input,
            num_outputs=config.size,
            weights_initializer=tf.constant_initializer(0.5),
            biases_initializer=tf.constant_initializer(0.1),
            activation_fn=None)

        self.feed_list = [input]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    test_main(PDFC(), TFFC(), FCConfig())
