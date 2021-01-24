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


class MatmulConfig(APIConfig):
    def __init__(self):
        super(MatmulConfig, self).__init__("matmul")
        self.feed_spec = [{"range": [-1, 1]}, {"range": [-1, 1]}]


class PDMatmul(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)
        result = paddle.matmul(
            x=x,
            y=y,
            transpose_x=config.transpose_x,
            transpose_y=config.transpose_y)

        self.feed_list = [x, y]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, y])


class TorchMatmul(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)
        if config.transpose_x:
            rank_of_x = len(config.x_shape)
            x_transposed = torch.transpose(
                input=x, dim0=rank_of_x - 2, dim1=rank_of_x - 1)
        else:
            x_transposed = x
        if config.transpose_y:
            rank_of_y = len(config.y_shape)
            y_transposed = torch.transpose(
                input=y, dim0=rank_of_y - 2, dim1=rank_of_y - 1)
        else:
            y_transposed = y
        result = torch.matmul(input=x_transposed, other=y_transposed)

        self.feed_list = [x, y]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, y])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDMatmul(), torch_obj=TorchMatmul(), config=MatmulConfig())
