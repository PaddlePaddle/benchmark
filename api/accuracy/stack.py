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
from abs import run_and_check, feed_random_data 
      
class PaddleStack(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False):
        self.name = "stack"
        with fluid.program_guard(self.main_program, self.startup_program):
            data1 = fluid.data(
                name='data1', shape=[10, 10, 100, 100], dtype='float32', lod_level=0)
            data1.stop_gradient = False
            data2 = fluid.data(
                name='data2', shape=[10, 10, 100, 100], dtype='float32', lod_level=0)
            data2.stop_gradient = False
            result = fluid.layers.stack([data1, data2], axis=1)

            self.feed_vars = [data1, data2]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [data1, data2])


class TensorflowStack(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False):
        self.name = "stack"
        self.allow_growth = True

        data1 = tf.placeholder(name='data1', shape=[10, 10, 100, 100], dtype=tf.float32)
        data2 = tf.placeholder(name='data2', shape=[10, 10, 100, 100], dtype=tf.float32)
        result = tf.stack([data1, data2], axis=1)

        self.feed_list = [data1, data2]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [data1, data2])

def main(backward, use_gpu):
    pd_obj = PaddleStack()
    tf_obj = TensorflowStack()
    run_and_check(pd_obj, tf_obj, backward, use_gpu, name="stack")

if __name__ == '__main__':
    args = parse_args()
    main(backward=args.backward, use_gpu=args.use_gpu)
