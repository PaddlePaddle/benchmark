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
from common import api_param


class PDScale(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, config):
        import paddle.fluid as fluid

        self.name = "scale"
        with fluid.program_guard(self.main_program, self.startup_program):
            x = fluid.data(
                name='x',
                shape=config.x_shape,
                dtype=config.x_dtype,
                lod_level=0)
            x.stop_gradient = False
            result = fluid.layers.scale(
                x=x,
                scale=config.scale,
                bias=0.0,
                bias_after_scale=True,
                act=None)

            self.feed_vars = [x]
            self.fetch_vars = [result]


class TFScale(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        import tensorflow as tf

        self.name = "scale"
        self.allow_growth = True

        x = tf.placeholder(
            name='x', shape=config.x_shape, dtype=tf.as_dtype(config.x_dtype))
        result = tf.scalar_mul(scalar=config.scale, x=x)

        self.feed_list = [x]
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(PDScale(), TFScale(), config=api_param.APIConfig("scale", ""))
