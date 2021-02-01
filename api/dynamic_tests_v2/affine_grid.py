#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file ethetacept in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either ethetapress or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from common_import import *


class PDAffineGrid(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        theta = self.variable(
            name='theta', shape=config.theta_shape, dtype=config.theta_dtype)
        result = paddle.nn.functional.affine_grid(
            theta,
            out_shape=config.out_shape,
            align_corners=config.align_corners)

        self.feed_list = [theta]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [theta])


class TorchAffineGrid(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        theta = self.variable(
            name='theta', shape=config.theta_shape, dtype=config.theta_dtype)
        result = torch.nn.functional.affine_grid(
            theta, size=config.out_shape, align_corners=config.align_corners)

        self.feed_list = [theta]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [theta])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDAffineGrid(),
        torch_obj=TorchAffineGrid(),
        config=APIConfig("affine_grid"))
