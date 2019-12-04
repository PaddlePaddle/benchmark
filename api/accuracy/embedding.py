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
      
class PaddleEmbedding(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False):
        self.name = "embedding"
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data', shape=[10, 10], dtype='int64', lod_level=0)
            table = fluid.layers.create_parameter(
                shape=[64, 128], dtype='float32', name='table')
            result = fluid.embedding(input=data, size=[64, 128], param_attr='table')

            self.feed_vars = [data, table]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [table])


class TensorflowEmbedding(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False):
        self.name = "embedding"
        self.allow_growth = True

        data = tf.placeholder(name='data', shape=[10, 10], dtype=tf.int64)
        table = tf.placeholder(name='table', shape=[64, 128], dtype=tf.float32)
        result = tf.nn.embedding_lookup(ids=data, params=table, max_norm=None)

        self.feed_list = [data, table]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [table])


def feed_random_data(pd_obj, tf_obj):
    assert len(pd_obj.feed_vars) == len(tf_obj.feed_list)

    pd_feed = {}
    tf_feed = {}
    for i in xrange(len(pd_obj.feed_vars)):
        pd_var = pd_obj.feed_vars[i]
        tf_var = tf_obj.feed_list[i]

        assert pd_var.shape == tf_var.shape
        assert pd_obj.convert_dtype(pd_var.dtype) == tf_obj.convert_dtype(tf_var.dtype)
        dtype = pd_obj.convert_dtype(pd_var.dtype)
        if dtype == "int64" or dtype == "int32":
            data = np.random.randint(0, 64, pd_var.shape).astype(dtype)
        else:
            data = np.random.random(pd_var.shape).astype(dtype)

        pd_feed[pd_var.name] = data
        tf_feed[tf_var] = data
    return pd_feed, tf_feed

def main(backward, use_gpu):
    # Define Paddle program
    pd_obj = PaddleEmbedding()
    pd_obj.build_program(backward=backward)

    # Define Tensorflow graph
    tf_obj = TensorflowEmbedding()
    tf_obj.build_graph(backward=backward)

    pd_feed, tf_feed = feed_random_data(pd_obj, tf_obj)

    # Run Paddle
    pd_outputs = pd_obj.run_with_executor(use_gpu=use_gpu, feed=pd_feed, check_output=False)

    # Run Tensorflow
    tf_outputs = tf_obj.run(use_gpu=use_gpu, feed=tf_feed, check_output=False)

    utils.check_outputs(pd_outputs, tf_outputs, name="embedding")

if __name__ == '__main__':
    args = parse_args()
    main(backward=args.backward, use_gpu=args.use_gpu)
