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


class EqualAllConfig(APIConfig):
    def __init__(self):
        super(EqualAllConfig, self).__init__("equal_all")
        self.run_tf = False


class PDEqualAll(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        data_x = self.variable(
            name='x', shape=config.x_shape, dtype=config.x_dtype)
        data_y = self.variable(
            name='y', shape=config.y_shape, dtype=config.y_dtype)
        out = paddle.equal_all(x=data_x, y=data_y)

        self.feed_vars = [data_x, data_y]
        self.fetch_vars = [out]


if __name__ == '__main__':
    test_main(PDEqualAll(), config=EqualAllConfig())
