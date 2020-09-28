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


class Relu6Config(APIConfig):
    def __init__(self):
        super(Relu6Config, self).__init__("relu6")
        self.feed_spec = {"range": [-1, 1]}
        # relu6 belongs to activation op series which only has one variable
        # thus abs can reuse activation parameters 
        self.alias_config = APIConfig("activation")


class PDRelu6(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(
            name='x', shape=config.alias.x_shape, dtype=config.alias.x_dtype)
        out = paddle.nn.functional.relu6(x=x)

        self.feed_vars = [x]
        self.fetch_vars = [out]
        if config.backward:
            self.append_gradients(out, [x])


class TFRelu6(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(
            name='x', shape=config.alias.x_shape, dtype=config.alias.x_dtype)
        out = tf.nn.relu6(features=x)

        self.feed_list = [x]
        self.fetch_list = [out]
        if config.backward:
            self.append_gradients(out, [x])


if __name__ == '__main__':
    test_main(PDRelu6(), TFRelu6(), config=Relu6Config())
