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


class PDFetch(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name="x", shape=[1600000], dtype="float32")

        self.feed_vars = [x]
        self.fetch_vars = [x]


class TFFetch(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name="x", shape=[1600000], dtype="float32")

        self.feed_list = [x]
        self.fetch_list = [x]


if __name__ == '__main__':
    test_main_without_json(PDFetch(), TFFetch(), config=APIConfig("fetch"))
