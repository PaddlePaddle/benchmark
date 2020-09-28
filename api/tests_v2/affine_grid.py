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


class AffineGridConfig(APIConfig):
    def __init__(self):
        super(AffineGridConfig, self).__init__("affine_grid")
        self.run_tf = False


class PDAffineGrid(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        theta = self.variable(
            name='theta', shape=config.theta_shape, dtype=config.theta_dtype)
        out = paddle.nn.functional.affine_grid(
            theta,
            out_shape=config.out_shape,
            align_corners=config.align_corners)
        self.feed_vars = [theta]
        self.fetch_vars = [out]
        if config.backward:
            self.append_gradients(out, [theta])


if __name__ == '__main__':
    test_main(PDAffineGrid(), config=AffineGridConfig())
