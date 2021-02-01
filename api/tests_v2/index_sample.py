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


class IndexSampleConfig(APIConfig):
    def __init__(self):
        super(IndexSampleConfig, self).__init__("index_sample")
        self.run_tf = False

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(IndexSampleConfig, self).init_from_json(filename, config_id,
                                                      unknown_dim)
        self.feed_spec = [
            {
                "range": [0, 1]
            },  # x
            {
                "range": [0, self.x_shape[-1]]
            }  # index
        ]


class PDIndexSample(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        index = self.variable(
            name='index', shape=config.index_shape, dtype=config.index_dtype)
        result = paddle.index_sample(x=x, index=index)

        self.feed_vars = [x, index]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(pd_obj=PDIndexSample(), config=IndexSampleConfig())
