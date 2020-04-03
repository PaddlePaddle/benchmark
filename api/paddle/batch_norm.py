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

from main import test_main

import sys
sys.path.append("..")
from common import paddle_api_benchmark as paddle_api
from common import tensorflow_api_benchmark as tensorflow_api
      

class PDBatchNorm(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False, dtype=None):
        import paddle.fluid as fluid

        self.name = "batch_norm"
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data', shape=[10, 100, 100, 32], dtype='float32', lod_level=0)
            scale = fluid.layers.create_parameter(
                name='scale', shape=[32], dtype="float32")
            bias = fluid.layers.create_parameter(
                name='bias', shape=[32], dtype="float32")
            data.stop_gradient = False
            result = fluid.layers.batch_norm(input=data,
                                             act=None,
                                             is_test=False,
                                             momentum=0.9,
                                             epsilon=0.001,
                                             param_attr="scale",
                                             bias_attr="bias",
                                             data_layout="NHWC")

            self.feed_vars = [data, scale, bias]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [data, scale, bias])


class TFBatchNorm(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False, dtype=None):
        import tensorflow as tf

        self.name = "batch_norm"
        self.allow_growth = True

        data = tf.placeholder(name='data', shape=[10, 100, 100, 32], dtype=tf.float32)
        scale = tf.placeholder(name='scale', shape=[32], dtype=tf.float32)
        bias = tf.placeholder(name='bias', shape=[32], dtype=tf.float32)
        mean, var = tf.nn.moments(x=data, axes=[0, 1, 2], shift=None, keepdims=False)
        result = tf.nn.batch_normalization(x=data,
                                           mean=mean,
                                           variance=var,
                                           offset=bias,
                                           scale=scale,
                                           variance_epsilon=0.001)

        self.feed_list = [data, scale, bias]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [data, scale, bias])


if __name__ == '__main__':
    test_main(PDBatchNorm(), TFBatchNorm(), feed_spec=None)
