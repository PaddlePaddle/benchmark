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


class PDFillConstant(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        result = paddle.fluid.layers.fill_constant(
            shape=config.shape, dtype=config.dtype, value=config.value)

        self.feed_list = []
        self.fetch_list = [result]


class TorchConstant(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        torch_list = []
        torch_tensor = torch.tensor(config.shape, dtype=torch.float32)
        result = torch.nn.init.constant_(tensor=torch_tensor, val=config.value)

        self.feed_list = []
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDFillConstant(),
        torch_obj=TorchConstant(),
        config=APIConfig("fill_constant"))
