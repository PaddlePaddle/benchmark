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
      
class PaddleElementwiseMul(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False):
        self.name = "elementwise_mul"
        with fluid.program_guard(self.main_program, self.startup_program):
            x = fluid.data(
                name='x', shape=[1, 32, 10, 768], dtype='float32', lod_level=0)
            y = fluid.data(
                name='y', shape=[10, 768], dtype='float32', lod_level=0)
            x.stop_gradient = False
            y.stop_gradient = False
            result = fluid.layers.elementwise_mul(x=x,
                                                  y=y,
                                                  axis=-1,
                                                  act=None)

            self.feed_vars = [x, y]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [x, y])


class TensorflowElementwiseMul(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False):
        self.name = "elementwise_mul"
        self.allow_growth = True

        x = tf.placeholder(name='x', shape=[1, 32, 10, 768], dtype=tf.float32)
        y = tf.placeholder(name='y', shape=[10, 768], dtype=tf.float32)
        result = tf.multiply(x=x, y=y)

        self.feed_list = [x, y]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [x, y])


def main(backward, use_gpu):
    pd_obj = PaddleElementwiseMul()
    tf_obj = TensorflowElementwiseMul()
    run_and_check(pd_obj, tf_obj, backward, use_gpu, name="elementwise_mul")

if __name__ == '__main__':
    args = parse_args()
    main(backward=args.backward, use_gpu=args.use_gpu)
