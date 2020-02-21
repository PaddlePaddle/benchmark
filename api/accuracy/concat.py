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
      
class PaddleConcat(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False):
        self.name = "concat"
        with fluid.program_guard(self.main_program, self.startup_program):
            data1 = fluid.data(
                name='data1', shape=[100, 200], dtype='float32', lod_level=0)
            data2 = fluid.data(
                name='data2', shape=[100, 200], dtype='float32', lod_level=0)
            data1.stop_gradient = False
            data2.stop_gradient = False
            result = fluid.layers.concat([data1, data2], axis=0)

            self.feed_vars = [data1, data2]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [data1, data2])


class TensorflowConcat(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False):
        self.name = "concat"
        self.allow_growth = True

        data1 = tf.placeholder(name='data1', shape=[100, 200], dtype=tf.float32)
        data2 = tf.placeholder(name='data2', shape=[100, 200], dtype=tf.float32)
        result = tf.concat([data1, data2], 0)

        self.feed_list = [data1, data2]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [data1, data2])


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
    pd_obj = PaddleConcat()
    tf_obj = TensorflowConcat()
    run_and_check(pd_obj, tf_obj, backward, use_gpu, name="concat")

if __name__ == '__main__':
    args = parse_args()
    main(backward=args.backward, use_gpu=args.use_gpu)
