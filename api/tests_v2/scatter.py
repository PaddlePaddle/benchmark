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


class ScatterConfig(APIConfig):
    def __init__(self):
        super(ScatterConfig, self).__init__('scatter')
        self.run_tf = False

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(ScatterConfig, self).init_from_json(filename, config_id,
                                                  unknown_dim)
        self.feed_spec = [
            {
                "range": [0, 10]
            },  # input
            {
                "range": [0, int(min(self.input_shape))]
            },  # index
            {
                "range": [0, 10]
            }  # update
        ]


class PDScatter(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(
            name='x', shape=config.input_shape, dtype=config.input_dtype)
        index = self.variable(
            name='index',
            shape=config.index_shape,
            dtype=config.index_dtype,
            stop_gradient=True)
        updates = self.variable(
            name='updates',
            shape=config.updates_shape,
            dtype=config.updates_dtype)
        result = paddle.scatter(x=x, index=index, updates=updates)

        self.feed_vars = [x, index, updates]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x, updates])


if __name__ == '__main__':
    test_main(PDScatter(), config=ScatterConfig())
