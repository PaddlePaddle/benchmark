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


class DivideConfig(APIConfig):
    def __init__(self):
        super(DivideConfig, self).__init__('divide')
        self.alias_config = APIConfig("elementwise")
        self.feed_spec = [{"range": [1, 3]}, {"range": [1, 3]}]
        self.alias.atol=0.2

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(DivideConfig, self).init_from_json(filename, config_id,
                                                      unknown_dim)
        if len(self.alias.x_shape) > len(self.alias.y_shape) and self.alias.y_shape != [1]:
            self.alias.y_shape = unsqueeze_short(
                short=self.alias.y_shape, long=self.alias.x_shape)
        elif len(self.alias.x_shape) < len(self.alias.y_shape) and self.alias.x_shape != [1]:
            self.x_shape_unsqueezed = unsqueeze_short(
                short=self.alias.x_shape, long=self.alias.y_shape)

class PDDivide(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.alias.x_shape, dtype=config.alias.x_dtype)
        y = self.variable(name='y', shape=config.alias.y_shape, dtype=config.alias.y_dtype)

        result = paddle.divide(x=x, y=y)

        self.feed_vars = [x, y]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x, y])


class TFDivide(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.alias.x_shape, dtype=config.alias.x_dtype)
        y = self.variable(name='y', shape=config.alias.y_shape, dtype=config.alias.y_dtype)

        result = tf.math.divide(x=x, y=y)

        self.feed_list = [x, y]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, y])


if __name__ == '__main__':
    test_main(PDDivide(), TFDivide(), config=DivideConfig())
