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

from common_import import *


class PDNull(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=[1], dtype="float32")

        self.feed_vars = [x]
        self.fetch_vars = None


class TFNull(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=[1], dtype="float32")
        result = tf.identity(x)

        self.feed_list = [x]
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main_without_json(PDNull(), TFNull(), config=APIConfig("null"))
