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


class RemainderConfig(APIConfig):
    def __init__(self):
        super(RemainderConfig, self).__init__("remainder")
        self.feed_spec = [{"range": [-1000, 1000]}, {"range": [1, 1000]}]
        # abs belongs to activation op series which only has one parameter
        # thus abs can reuse activation.json. 
        self.alias_name = "elementwise"

    def disabled(self):
        return True if self.x_dtype == "float16" else False

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(RemainderConfig, self).init_from_json(filename, config_id,
                                                    unknown_dim)
        if len(self.x_shape) > len(self.y_shape) and self.y_shape != [1]:
            self.y_shape = unsqueeze_short(
                short=self.y_shape, long=self.x_shape)
        elif len(self.x_shape) < len(self.y_shape) and self.x_shape != [1]:
            self.x_shape = unsqueeze_short(
                short=self.x_shape, long=self.y_shape)


class PDRemainder(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)
        result = paddle.remainder(x=x, y=y)

        self.feed_vars = [x, y]
        self.fetch_vars = [result]


class TFRemainder(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)

        result = tf.math.floormod(x=x, y=y)

        self.feed_list = [x, y]
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(PDRemainder(), TFRemainder(), config=RemainderConfig())
