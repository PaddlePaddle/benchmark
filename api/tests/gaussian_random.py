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


@benchmark_registry.register("gaussian_random")
class GaussianRandomConfig(APIConfig):
    def __init__(self):
        super(GaussianRandomConfig, self).__init__("gaussian_random")
        self.run_torch = False
        self.feed_spec = [{"range": [-1, 1]}, {"range": [-1, 1]}]


@benchmark_registry.register("gaussian_random")
class PaddleGaussianRandom(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        result = paddle.fluid.layers.gaussian_random(
            shape=config.shape,
            mean=config.mean,
            std=config.std,
            seed=config.seed,
            dtype=config.dtype)

        self.feed_list = []
        self.fetch_list = [result]
