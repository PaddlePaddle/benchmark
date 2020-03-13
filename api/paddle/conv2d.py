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

from main import test_speed_main
import paddle.fluid as fluid

import sys
sys.path.append("..")
from common import paddle_api_benchmark as paddle_api

class conv2d(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False, dtype=None):
        with fluid.program_guard(self.main_program, self.startup_program):
            input = fluid.data(
                name='input', shape=[1, 1, 80, 1008], dtype=dtype, lod_level=0)
            filters = fluid.layers.create_parameter(
                name='filters', shape=[1, 1, 3, 32], dtype=dtype)
            input.stop_gradient = False
            result = fluid.layers.conv2d(
                input=input,
                num_filters=1,
                filter_size=(3, 32),
                padding=(1, 8),
                stride=(1, 16),
                param_attr='filters',
                bias_attr=False,
                use_cudnn=True)

            self.feed_vars = [input, filters]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [input, filters])


if __name__ == '__main__':
    test_speed_main(conv2d())
