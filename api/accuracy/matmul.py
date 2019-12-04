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

from abs import feed_random_data

import sys
sys.path.append("..")
from common import paddle_api_benchmark as paddle_api
from common import tensorflow_api_benchmark as tensorflow_api
from common import utils
      
class PaddleMatmul(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False):
        self.name = "matmul"
        with fluid.program_guard(self.main_program, self.startup_program):
            x = fluid.data(
                name='x', shape=[32, 128, 768], dtype='float32', lod_level=0)
            y = fluid.data(
                name='y', shape=[32, 768, 256], dtype='float32', lod_level=0)
            x.stop_gradient = False
            y.stop_gradient = False
            result = fluid.layers.matmul(x=x,
                                         y=y,
                                         transpose_x=False,
                                         transpose_y=False,
                                         alpha=1.0)

            self.feed_vars = [x, y]
            if backward:
                gradients = fluid.backward.calc_gradient(result, [x, y])
                self.fetch_vars = [result, gradients[0], gradients[1]]
            else:
                self.fetch_vars = [result]


class TensorflowMatmul(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False):
        self.name = "matmul"
        self.allow_growth = True

        x = tf.placeholder(name='x', shape=[32, 128, 768], dtype=tf.float32)
        y = tf.placeholder(name='y', shape=[32, 768, 256], dtype=tf.float32)
        result = tf.matmul(a=x,
                           b=y,
                           transpose_a=False,
                           transpose_b=False,
                           adjoint_a=False,
                           adjoint_b=False,
                           a_is_sparse=False,
                           b_is_sparse=False)

        self.feed_list = [x, y]
        if backward:
            gradients = tf.gradients(result, [x, y])
            self.fetch_list = [result, gradients[0], gradients[1]]
        else:
            self.fetch_list = [result]


def main(backward, use_gpu):
    # Define Paddle program
    pd_obj = PaddleMatmul()
    pd_obj.build_program(backward=backward)

    # Define Tensorflow graph
    tf_obj = TensorflowMatmul()
    tf_obj.build_graph(backward=backward)

    pd_feed, tf_feed = feed_random_data(pd_obj, tf_obj)

    # Run Paddle
    pd_outputs = pd_obj.run_with_executor(use_gpu=use_gpu, feed=pd_feed, check_output=False)

    # Run Tensorflow
    tf_outputs = tf_obj.run(use_gpu=use_gpu, feed=tf_feed, check_output=False)

    utils.check_outputs(pd_outputs, tf_outputs, name="matmul")

if __name__ == '__main__':
    main(backward=False, use_gpu=True)
