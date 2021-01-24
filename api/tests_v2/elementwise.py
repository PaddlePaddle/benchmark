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


class ElementwiseConfig(APIConfig):
    def __init__(self, op_type="elementwise"):
        super(ElementwiseConfig, self).__init__(op_type)
        self.api_name = 'add'
        self.api_list = {
            'add': 'add',
            'maximum': 'maximum',
            'minimum': 'minimum',
            'multiply': 'multiply',
            'pow': 'pow',
            'subtract': 'subtract'
        }
        self.feed_spec = [{"range": [-1, 1]}, {"range": [-1, 1]}]

    def disabled(self):
        if self.api_name in ["pow", "maximum", "minimum", "divide"
                             ] and self.x_dtype == "float16":
            print(
                "Warning:\n"
                "  1. This config is disabled because float16 is not supported for %s.\n"
                % (self.api_name))
            return True
        return super(ElementwiseConfig, self).disabled()

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(ElementwiseConfig, self).init_from_json(filename, config_id,
                                                      unknown_dim)
        if len(self.x_shape) > len(self.y_shape) and self.y_shape != [1]:
            self.y_shape = unsqueeze_short(
                short=self.y_shape, long=self.x_shape)
        elif len(self.x_shape) < len(self.y_shape) and self.x_shape != [1]:
            self.x_shape_unsqueezed = unsqueeze_short(
                short=self.x_shape, long=self.y_shape)


class PDElementwise(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)

        result = self.layers(config.api_name, x=x, y=y)

        self.feed_vars = [x, y]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x, y])


class TFElementwise(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)

        result = self.layers(config.api_name, x=x, y=y)

        self.feed_list = [x, y]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, y])


if __name__ == '__main__':
    test_main(PDElementwise(), TFElementwise(), config=ElementwiseConfig())
