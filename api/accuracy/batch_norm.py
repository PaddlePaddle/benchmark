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

import sys
sys.path.append("..")
from common import paddle_api_benchmark as paddle_api
from common import tensorflow_api_benchmark as tensorflow_api
from common import utils
from abs import feed_random_data, run_and_check 
      
class PaddleBatchNorm(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False):
        self.name = "batch_norm"
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data', shape=[10, 10, 100, 100], dtype='float32', lod_level=0)
            param_attr = fluid.ParamAttr(name='batch_norm_w', initializer=fluid.initializer.Constant(value=1.0))
            bias_attr = fluid.ParamAttr(name='batch_norm_b', initializer=fluid.initializer.Constant(value=0.0))
            data.stop_gradient = False
            result = fluid.layers.batch_norm(input=data, param_attr = param_attr, bias_attr = bias_attr, epsilon=0.001)

            self.feed_vars = [data]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [data])


class TensorflowBatchNorm(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False):
        self.name = "batch_norm"
        self.allow_growth = True

        data = tf.placeholder(name='data', shape=[10, 10, 100, 100], dtype=tf.float32)
        img_shape = [10, 10, 100, 100]
        size = 100
        axis = list(range(len(img_shape) - 1))
        mean, var = tf.nn.moments(data, axis)
        scale = tf.Variable(tf.ones([size]))
        shift = tf.Variable(tf.zeros([size]))
        epsilon = 0.001 
        result = tf.nn.batch_normalization(data, mean, var, shift, scale, epsilon)

        self.feed_list = [data]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [data])

def main(backward, use_gpu):
    pd_obj = PaddleBatchNorm()
    tf_obj = TensorflowBatchNorm()
    run_and_check(pd_obj, tf_obj, backward, use_gpu, name="batch_norm")

if __name__ == '__main__':
    args = parse_args()
    main(backward=args.backward, use_gpu=args.use_gpu)
