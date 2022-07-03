#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


@benchmark_registry.register("set_value")
class SetValueConfig(APIConfig):
    def __init__(self):
        super(SetValueConfig, self).__init__("set_value")
        self.run_torch = False

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(SetValueConfig, self).init_from_json(filename, config_id,
                                                   unknown_dim)


@benchmark_registry.register("set_value")
class PaddleSetValue(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        input.stop_gradient = True

        if config.is_tensor_value:
            value = self.variable(
                name='value',
                shape=config.value_shape,
                dtype=config.value_dtype)
            input[:, 10:500:2] = value

            self.feed_list = [input, value]
            self.fetch_list = [input]

        else:
            input[:, 0:20, ::2] = 10000

            self.feed_list = [input]
            self.fetch_list = [input]
