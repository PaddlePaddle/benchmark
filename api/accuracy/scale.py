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
      
class PaddleScale(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False):
        self.name = "scale"
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data', shape=[32, 128], dtype='float32', lod_level=0)
            data.stop_gradient = False
            result = fluid.layers.scale(x=data,
                                        scale=2.0,
                                        bias=0.0,
                                        bias_after_scale=True,
                                        act=None)

            self.feed_vars = [data]
            self.fetch_vars = [result]


class TensorflowScale(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False):
        self.name = "scale"
        self.allow_growth = True

        data = tf.placeholder(name='data', shape=[32, 128], dtype=tf.float32)
        result = tf.scalar_mul(scalar=2.0, x=data)

        self.feed_list = [data]
        self.fetch_list = [result]


def main(backward, use_gpu):
    pd_obj = PaddleScale()
    tf_obj = TensorflowScale()
    run_and_check(pd_obj, tf_obj, backward, use_gpu, name="scale")

if __name__ == '__main__':
    args = parse_args()
    main(backward=args.backward, use_gpu=args.use_gpu)
