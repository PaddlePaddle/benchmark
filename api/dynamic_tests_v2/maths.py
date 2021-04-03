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


class MathsConfig(APIConfig):
    def __init__(self):
        super(MathsConfig, self).__init__('maths')
        self.api_name = 'cos'
        self.api_list = {
            'cos': 'cos',
            'exp': 'exp',
            'log': 'log',
            'sin': 'sin',
            'sinh': 'sinh',
            'sqrt': 'sqrt',
            'square': 'square',
            'tanh': 'tanh'
        }
        self.alias_name = "activation"

    def disabled(self):
        if self.api_name in ["log"] and self.x_dtype == "float16":
            print(
                "Warning:\n"
                "  1. This config is disabled because float16 is not supported for %s.\n"
                % (self.api_name))
            return True
        return super(MathsConfig, self).disabled()


class PDMaths(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = self.layers(config.api_name, x=x)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TorchMaths(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = self.layers(config.api_name, input=x)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDMaths(), torch_obj=TorchMaths(), config=MathsConfig())
