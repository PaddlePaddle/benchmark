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


class LinspaceConfig(APIConfig):
    def __init__(self):
        super(LinspaceConfig, self).__init__("linspace")
        self.feed_spec = [
                {"range": [0, 100] }, #start
                {"range": [200, 300] }  #stop
                ]


class PDLinspace(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        start = self.variable(
            name="start", shape=[1], dtype=config.start_dtype)
        stop = self.variable(
            name="stop", shape=[1], dtype=config.stop_dtype)
        result = paddle.linspace(start=start, stop=stop, num=config.num, dtype=config.dtype)

        self.feed_vars = [start, stop]
        self.fetch_vars = [result]


class TFLinspace(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        start = self.variable(
            name="start", shape=[1], dtype=config.start_dtype)
        stop = self.variable(
            name="stop", shape=[1], dtype=config.stop_dtype)
        result = tf.linspace(start=start, stop=stop, num=config.num)

        self.feed_list = [start, stop]
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(PDLinspace(), TFLinspace(), config=LinspaceConfig())
