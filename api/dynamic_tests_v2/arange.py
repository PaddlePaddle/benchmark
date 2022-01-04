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


class PDArange(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        result = paddle.arange(
            start=config.start,
            end=config.end,
            step=config.step,
            dtype=config.dtype)

        self.feed_list = []
        self.fetch_list = [result]


class TorchArange(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        result = torch.range(
            start=config.start,
            end=config.end - config.step,
            step=config.step,
            dtype=config.dtype)

        self.feed_list = []
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDArange(),
        torch_obj=TorchArange(),
        config=APIConfig("arange"))
