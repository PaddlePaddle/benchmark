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


class PDGridSample(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        grid = self.variable(
            name='grid', shape=config.grid_shape, dtype=config.grid_dtype)
        out = paddle.nn.functional.grid_sample(
            x,
            grid,
            mode=config.mode,
            padding_mode=config.padding_mode,
            align_corners=config.align_corners)
        self.feed_list = [x, grid]
        self.fetch_list = [out]
        if config.backward:
            self.append_gradients(out, [x, grid])


class TorchGridSample(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(
            name='input', shape=config.x_shape, dtype=config.x_dtype)
        grid = self.variable(
            name='grid', shape=config.grid_shape, dtype=config.grid_dtype)
        out = torch.nn.functional.grid_sample(
            x,
            grid,
            mode=config.mode,
            padding_mode=config.padding_mode,
            align_corners=config.align_corners)
        self.feed_list = [x, grid]
        self.fetch_list = [out]
        if config.backward:
            self.append_gradients(out, [x, grid])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDGridSample(),
        torch_obj=TorchGridSample(),
        config=APIConfig('grid_sample'))
