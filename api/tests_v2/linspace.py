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


class PDLinspace(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        result = paddle.linspace(start=config.start, stop=config.stop, num=config.num, dtype=config.dtype)

        self.feed_vars = []
        self.fetch_vars = [result]


class TFLinspace(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        result = tf.linspace(start=config.start, stop=config.stop, num=config.num)

        self.feed_list = []
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(PDLinspace(), TFLinspace(), config=APIConfig('linspace'))
