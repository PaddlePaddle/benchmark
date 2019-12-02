#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from args import parse_args
import paddle.fluid as fluid

import sys
sys.path.append("..")
from common import paddle_api_benchmark as paddle_api

class abs(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data', shape=[10, 10, 100, 100], dtype='float32', lod_level=0)
            out = fluid.layers.abs(x=data)

            self.feed_vars = [data]
            self.fetch_vars = [out]
        #print(self.main_program)


if __name__ == '__main__':
    args = parse_args()
    obj = abs()
    obj.build_program(backward=args.backward)
    if args.run_with_executor:
        obj.run_with_executor(use_gpu=args.use_gpu,
                              repeat=args.repeat,
                              log_level=args.log_level,
                              check_output=args.check_output,
                              profiler=args.profiler)
    else:
        obj.run_with_core_executor(use_gpu=args.use_gpu,
                                   repeat=args.repeat,
                                   log_level=args.log_level,
                                   check_output=args.check_output,
                                   profiler=args.profiler)
