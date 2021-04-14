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


class AnyConfig(APIConfig):
    def __init__(self):
        super(AnyConfig, self).__init__('any')
        self.alias_name = "reduce"

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(AnyConfig, self).init_from_json(filename, config_id, unknown_dim)
        self.x_dtype = "bool"
        # Update the parameters information. It will be removed and reimplemented
        # in api_param.py.
        for var in self.variable_list:
            if var.type == "Variable" and var.name == "x":
                var.dtype = "bool"


class PDAny(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = paddle.any(x=x, axis=config.axis, keepdim=config.keepdim)

        self.feed_vars = [x]
        self.fetch_vars = [result]


class TFAny(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = tf.math.reduce_any(
            input_tensor=x, axis=config.axis, keepdims=config.keepdim)

        self.feed_list = [x]
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(PDAny(), TFAny(), config=AnyConfig())
