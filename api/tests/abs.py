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


class AbsConfig(api_param.APIConfig):
    def __init__(self):
        super(AbsConfig, self).__init__('abs', '')
        self.feed_spec = {"range": [-1, 1]}


class PDAbs(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, config):
        import paddle.fluid as fluid

        self.name = "abs"
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data',
                shape=config.x_shape,
                dtype=config.x_dtype,
                lod_level=0)
            data.stop_gradient = False
            result = fluid.layers.abs(x=data)

            self.feed_vars = [data]
            self.fetch_vars = [result]
            if config.backward:
                self.append_gradients(result, [data])


class TFAbs(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        import tensorflow as tf

        self.name = "abs"
        self.allow_growth = True

        data = tf.placeholder(
            name='data',
            shape=config.x_shape,
            dtype=tf.as_dtype(config.x_dtype))
        result = tf.abs(x=data)

        self.feed_list = [data]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [data])


if __name__ == '__main__':
    test_main(PDAbs(), TFAbs(), config=AbsConfig())
