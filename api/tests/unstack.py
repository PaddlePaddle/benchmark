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


@benchmark_registry.register("unstack")
class UnstackConfig(APIConfig):
    def __init__(self):
        super(UnstackConfig, self).__init__('unstack')
        self.run_torch = False
        print("[WARNING]: Pytorch dosen`t support unstack currently.")


@benchmark_registry.register("unstack")
class PaddleUnStack(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = paddle.unstack(x=x, axis=config.axis)

        self.feed_list = [x]
        if type(result) == list:
            self.fetch_list = result
            # computing a list of gradients is not supported
            config.backward = False
        else:
            self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])
