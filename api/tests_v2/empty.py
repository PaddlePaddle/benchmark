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


class EmptyConfig(APIConfig):
    def __init__(self):
        super(EmptyConfig, self).__init__('empty')
        self.run_tf = False


class PDEmpty(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        result = paddle.empty(shape=config.shape, dtype=config.dtype)

        self.feed_vars = []
        self.fetch_vars = [result]


if __name__ == '__main__':
    test_main(PDEmpty(), config=EmptyConfig())
