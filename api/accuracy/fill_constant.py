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
      
class PaddleFillConstant(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False):
        self.name = "fill_constant"
        with fluid.program_guard(self.main_program, self.startup_program):
            result = fluid.layers.fill_constant(shape=[10, 10, 100, 100],
                                                dtype='int64',
                                                value=3)

            self.feed_vars = []
            self.fetch_vars = [result]


class TensorflowFillConstant(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False):
        self.name = "fill_constant"
        self.allow_growth = True

        result = tf.constant(shape=[10, 10, 100, 100],
                             dtype=tf.int64,
                             value=3)

        self.feed_list = []
        self.fetch_list = [result]


def main(backward, use_gpu):
    pd_obj = PaddleFillConstant()
    tf_obj = TensorflowFillConstant()
    run_and_check(pd_obj, tf_obj, backward, use_gpu, name="fill_constant")

if __name__ == '__main__':
    args = parse_args()
    main(backward=args.backward, use_gpu=args.use_gpu)
