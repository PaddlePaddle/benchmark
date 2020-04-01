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

from main import test_main

import sys
sys.path.append("..")
from common import paddle_api_benchmark as paddle_api


class PDElementwiseMul(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False, dtype=None):
        import paddle.fluid as fluid

        self.name = "elementwise_mul"
        with fluid.program_guard(self.main_program, self.startup_program):
            x = fluid.data(
                name='x', shape=[50, 128, 1000], dtype='float32', lod_level=0)
            y = fluid.data(
                name='y', shape=[1, 128, 1000], dtype='float32', lod_level=0)
            x.stop_gradient = False
            y.stop_gradient = False
            result = fluid.layers.elementwise_mul(x, y)

            self.feed_vars = [x, y]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [x, y])


if __name__ == '__main__':
    test_main(PDElementwiseMul(), feed_spec=None)
