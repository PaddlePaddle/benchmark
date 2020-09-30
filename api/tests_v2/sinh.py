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


class SinhConfig(APIConfig):
    def __init__(self):
        super(SinhConfig, self).__init__("sinh")
        self.feed_spec = {"range": [-1, 1]}
        # sinh belongs to activation op series which only has one variable
        # thus sinh can reuse activation parameters
        self.alias_config = APIConfig("activation")


class PDSinh(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(
            name="x", shape=config.alias.x_shape, dtype=config.alias.x_dtype)
        result = paddle.sinh(x=x)

        self.feed_vars = [x]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TFSinh(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(
            name='x', shape=config.alias.x_shape, dtype=config.alias.x_dtype)
        result = tf.math.sinh(x=x)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(PDSinh(), TFSinh(), config=SinhConfig())
