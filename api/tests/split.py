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
      
class PDSplit(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False, dtype=None):
        import paddle.fluid as fluid

        self.name = "split"
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data', shape=[10, 10, 100, 100], dtype='float32', lod_level=0)
            data.stop_gradient = True
            result1, result2, result3 = fluid.layers.split(data,
                                                           num_or_sections=[1, 2, 7],
                                                           dim=1)

            self.feed_vars = [data]
            self.fetch_vars = [result1, result2, result3]
            if backward:
                self.append_gradients([result1, result2, result3], [data])


class TFSplit(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False, dtype=None):
        import tensorflow as tf

        self.name = "split"
        self.allow_growth = True

        data = tf.placeholder(name='data', shape=[10, 10, 100, 100], dtype=tf.float32)
        result1, result2, result3 = tf.split(value=data,
                                             num_or_size_splits=[1, 2, 7],
                                             axis=1)

        self.feed_list = [data]
        self.fetch_list = [result1, result2, result3]
        if backward:
            self.append_gradients([result1, result2, result3], [data])


if __name__ == '__main__':
    test_main(PDSplit(), TFSplit(), feed_spec=None)
