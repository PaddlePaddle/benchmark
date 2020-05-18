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


class ArgConfig(APIConfig):
    def __init__(self):
        super(ArgConfig, self).__init__('arg')
        self.api = 'max'
        self.api_list = {'max': 'argmax', 'min': 'argmin'}

    def to_tensorflow(self):
        self.tf_api = self.api_list[self.api]
        return self


class PDArg(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            x = fluid.data(
                name='x',
                shape=config.x_shape,
                dtype=config.x_dtype,
                lod_level=0)
            x.stop_gradient = False
            self.name = 'arg' + config.api
            result = self.layers(
                "arg" + config.api,
                x=x,
                axis=config.axis, )

            self.feed_vars = [x]
            self.fetch_vars = [result]
            if config.backward:
                self.append_gradients(result, [x])


class TFArg(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.placeholder(
            name='x', shape=config.x_shape, dtype=config.x_dtype)
        self.name = config.tf_api
        result = self.layers(".math", config.tf_api, input=x, axis=config.axis)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(PDArg(), TFArg(), ArgConfig())
