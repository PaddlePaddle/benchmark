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


@benchmark_registry.register("put_along_axis")
class PutAlongAxisConfig(APIConfig):
    def __init__(self):
        super(PutAlongAxisConfig, self).__init__("put_along_axis")
        self.feed_spec = [{"range": [-1, 1]}, {"range": [0, 100]}]
        self.run_torch = False


@benchmark_registry.register("put_along_axis")
class PaddlePutAlongAxis(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        arr = self.variable(
            name="arr", shape=config.arr_shape, dtype=config.arr_dtype)
        indices = self.variable(
            name="indices",
            shape=config.indices_shape,
            dtype=config.indices_dtype)
        result = paddle.put_along_axis(
            arr=arr, indices=indices, values=config.values, axis=config.axis)

        self.feed_list = [arr]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [arr])
