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


class SoftplusConfig(APIConfig):
    def __init__(self):
        super(SoftplusConfig, self).__init__("softplus")
        self.feed_spec = {"range": [-1, 1]}
        # softplus belongs to activation op series which only has one variable
        # thus abs can reuse activation parameters 
        self.alias_config = APIConfig("activation")


class PDSoftplus(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(
            name='x', shape=config.alias.x_shape, dtype=config.alias.x_dtype)
        out = paddle.nn.functional.softplus(x=x)

        self.feed_vars = [x]
        self.fetch_vars = [out]
        if config.backward:
            self.append_gradients(out, [x])


class TFSoftplus(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(
            name='x', shape=config.alias.x_shape, dtype=config.alias.x_dtype)
        out = tf.math.softplus(features=x)

        self.feed_list = [x]
        self.fetch_list = [out]
        if config.backward:
            self.append_gradients(out, [x])


if __name__ == '__main__':
    test_main(PDSoftplus(), TFSoftplus(), config=SoftplusConfig())
