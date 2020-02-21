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

from __future__ import print_function

import os
os.environ["FLAGS_fraction_of_gpu_memory_to_use"] = "0.01"

import paddle.fluid as fluid
import tensorflow as tf
import numpy as np

from args import parse_args
from abs import feed_random_data, run_and_check

import sys
sys.path.append("..")
from common import paddle_api_benchmark as paddle_api
from common import tensorflow_api_benchmark as tensorflow_api
from common import utils
      
class PaddleFC(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False):
        self.name = "fc"
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data', shape=[32,32], dtype='float32', lod_level=0)
            data.stop_gradient = False
            #w = fluid.layers.create_parameter(
            #    shape=[32, 1000], dtype='float32', name='filters')
            w_param_attrs = fluid.ParamAttr(name="fc_weight",
                                initializer=fluid.initializer.ConstantInitializer(0.5),
                                learning_rate=1,
                                trainable=True)
            b_param_attrs = fluid.ParamAttr(name="fc_bias",
                                initializer=fluid.initializer.ConstantInitializer(0.1),
                                learning_rate=1,
                                trainable=True)
            result = fluid.layers.fc(input=data, size=1000, act='tanh', param_attr=w_param_attrs,
                                     bias_attr=b_param_attrs)

            self.feed_vars = [data]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [data])


class TensorflowFC(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False):
        self.name = "fully_connected"
        self.allow_growth = True

        data = tf.placeholder(name='data', shape=[32,32], dtype=tf.float32)
        #w = tf.placeholder(name='w', shape=[1000, 32], dtype=tf.float32)

        result = tf.contrib.layers.fully_connected(inputs=data, num_outputs=1000, activation_fn=tf.nn.tanh,
                                                   weights_initializer=tf.constant_initializer(0.5),
                                                   biases_initializer=tf.constant_initializer(0.1))

        self.feed_list = [data]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [data])


def feed_random_data(pd_obj, tf_obj):
    assert len(pd_obj.feed_vars) == len(tf_obj.feed_list)

    pd_feed = {}
    tf_feed = {}
    for i in xrange(len(pd_obj.feed_vars)):
        pd_var = pd_obj.feed_vars[i]
        tf_var = tf_obj.feed_list[i]

        assert pd_var.shape == tf_var.shape
        assert pd_obj.convert_dtype(pd_var.dtype) == tf_obj.convert_dtype(tf_var.dtype)
        data = np.random.random(pd_var.shape).astype(pd_obj.convert_dtype(pd_var.dtype))

        pd_feed[pd_var.name] = data
        tf_feed[tf_var] = data
    return pd_feed, tf_feed

def run_and_check(pd_obj, tf_obj, backward, use_gpu, name):
    # Define Paddle program
    pd_obj.build_program(backward=backward)

    # Define Tensorflow graph
    tf_obj.build_graph(backward=backward)

    pd_feed, tf_feed = feed_random_data(pd_obj, tf_obj)

    # Run Paddle
    pd_outputs = pd_obj.run_with_executor(use_gpu=use_gpu, feed=pd_feed, check_output=False)

    # Run Tensorflow
    tf_outputs = tf_obj.run(use_gpu=use_gpu, feed=tf_feed, check_output=False)

    utils.check_outputs(pd_outputs, tf_outputs, name=name)

def main(backward, use_gpu):
    pd_obj = PaddleFC()
    tf_obj = TensorflowFC()


    run_and_check(pd_obj, tf_obj, backward, use_gpu, name="fc")

if __name__ == '__main__':
    args = parse_args()
    main(backward=args.backward, use_gpu=args.use_gpu)
