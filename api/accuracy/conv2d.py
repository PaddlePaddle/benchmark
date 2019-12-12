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
      
class PaddleConv2d(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False):
        self.name = "conv2d"
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='input', shape=[32, 3, 224, 224], dtype='float32', lod_level=0)
            data.stop_gradient = False
            filters = fluid.layers.create_parameter(
                shape=[64, 3, 7, 7], dtype='float32', name='filters')
            result = fluid.layers.conv2d(input=data,
                                         num_filters=64,
                                         filter_size=[7, 7],
                                         stride=[2, 2],
                                         padding="SAME",
                                         dilation=[1, 1],
                                         groups=1,
                                         param_attr='filters',
                                         bias_attr=False,
                                         use_cudnn=True,
                                         act=None,
                                         data_format='NCHW')

            self.feed_vars = [data, filters]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [data, filters])


class TensorflowConv2d(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False):
        self.name = "conv2d"
        self.allow_growth = True

        data = tf.placeholder(name='input', shape=[32, 3, 224, 224], dtype=tf.float32)
        filters = tf.placeholder(name='filters', shape=[7, 7, 3, 64], dtype=tf.float32)
        result = tf.nn.conv2d(input=data,
                              filter=filters,
                              strides=[1, 1, 2, 2],
                              padding="SAME",
                              use_cudnn_on_gpu=True,
                              data_format='NCHW',
                              dilations=[1, 1, 1, 1])

        self.feed_list = [data, filters]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [data, filters])


def feed_random_data(pd_obj, tf_obj):
    assert len(pd_obj.feed_vars) == len(tf_obj.feed_list)

    pd_feed = {}
    tf_feed = {}
    for i in xrange(len(pd_obj.feed_vars)):
        pd_var = pd_obj.feed_vars[i]
        tf_var = tf_obj.feed_list[i]

        assert pd_obj.convert_dtype(pd_var.dtype) == tf_obj.convert_dtype(tf_var.dtype)
        data = np.random.random(pd_var.shape).astype(pd_obj.convert_dtype(pd_var.dtype))

        pd_feed[pd_var.name] = data
        if pd_var.shape == tf_var.shape:
            tf_feed[tf_var] = data
        else:
            # filters need to transpose
            tf_feed[tf_var] = np.transpose(data, (2, 3, 1, 0))
    return pd_feed, tf_feed

def main(backward, use_gpu):
    # Define Paddle program
    pd_obj = PaddleConv2d()
    pd_obj.build_program(backward=backward)

    # Define Tensorflow graph
    tf_obj = TensorflowConv2d()
    tf_obj.build_graph(backward=backward)

    pd_feed, tf_feed = feed_random_data(pd_obj, tf_obj)

    # Run Paddle
    pd_outputs = pd_obj.run_with_executor(use_gpu=use_gpu, feed=pd_feed, check_output=False)
    if backward and len(pd_outputs) >= 2:
        pd_outputs[2] = np.transpose(pd_outputs[2], (2, 3, 1, 0))

    # Run Tensorflow
    tf_outputs = tf_obj.run(use_gpu=use_gpu, feed=tf_feed, check_output=False)

    utils.check_outputs(pd_outputs, tf_outputs, name="conv2d")

if __name__ == '__main__':
    args = parse_args()
    main(backward=args.backward, use_gpu=args.use_gpu)
