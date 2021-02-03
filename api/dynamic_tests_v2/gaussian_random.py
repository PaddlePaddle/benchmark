#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


class PDGaussian_random(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        result = paddle.fluid.layers.gaussian_random(
            shape=config.shape,
            mean=config.mean,
            std=config.std)

        self.feed_list = []
        self.fetch_list = [result]


class TorchNormal(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        result = torch.normal(
            mean=config.mean,
            std=config.std,
            size=config.shape)

        self.feed_list = []
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDGaussian_random(),
        torch_obj=TorchNormal(),
        config=APIConfig("gaussian_random"))
