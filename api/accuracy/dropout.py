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
      
class PaddleDropout(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False):
        self.name = "dropout"
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data', shape=[10, 10, 100, 100], dtype='float32', lod_level=0)
            data.stop_gradient = False
            result = fluid.layers.dropout(x=data,
                                          dropout_prob=0.2,
                                          seed=123,
                                          dropout_implementation="upscale_in_train")

            self.feed_vars = [data]
            if backward:
                gradients = fluid.backward.calc_gradient(result, [data])
                self.fetch_vars = [result, gradients[0]]
            else:
                self.fetch_vars = [result]


class TensorflowDropout(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False):
        self.name = "dropout"
        self.allow_growth = True

        data = tf.placeholder(name='data', shape=[10, 10, 100, 100], dtype=tf.float32)
        result = tf.nn.dropout(x=data,
                               rate=0.2,
                               noise_shape=None,
                               seed=123)

        self.feed_list = [data]
        if backward:
            gradients = tf.gradients(result, [data])
            self.fetch_list = [result, gradients[0]]
        else:
            self.fetch_list = [result]


def main(backward, use_gpu):
    # Define Paddle program
    pd_obj = PaddleDropout()
    pd_obj.build_program(backward=backward)

    # Define Tensorflow graph
    tf_obj = TensorflowDropout()
    tf_obj.build_graph(backward=backward)

    pd_feed, tf_feed = feed_random_data(pd_obj, tf_obj)

    # Run Paddle
    pd_outputs = pd_obj.run_with_executor(use_gpu=use_gpu, feed=pd_feed, check_output=False)

    # Run Tensorflow
    tf_outputs = tf_obj.run(use_gpu=use_gpu, feed=tf_feed, check_output=False)

    utils.check_outputs(pd_outputs, tf_outputs, name="dropout")

if __name__ == '__main__':
    main(backward=False, use_gpu=True)
