#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from main import test_speed_main
import tensorflow as tf

import sys
sys.path.append("..")
from common import tensorflow_api_benchmark as tensorflow_api

class expand(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False):
        data = tf.placeholder(name='data', shape=[32, 807, 1], dtype=tf.float32)
        result = tf.tile(input=data, multiples=[4, 1, 807])

        self.feed_list = [data]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [data])


if __name__ == '__main__':
    test_speed_main(expand())
