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


class PDTranspose(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = paddle.transpose(x, config.perm)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])

    def compute_flop_and_byte(self, config):
        x_shape = config.x_shape
        forward_flop = 0
        forward_byte = 2 * numel(x_shape) * sizeof(config.x_dtype)
        if not config.backward:
            return forward_flop, forward_byte
        else:
            # To be implemented.
            return None, None


class TorchTranspose(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        if len(config.perm) == 2:
            result = torch.transpose(
                input=x, dim0=config.perm[0], dim1=config.perm[1]).contiguous()
        else:
            result = torch.permute(input=x, dims=config.perm).contiguous()

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDTranspose(),
        torch_obj=TorchTranspose(),
        config=APIConfig("transpose"))
