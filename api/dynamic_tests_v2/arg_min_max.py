#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class ArgMinMaxConfig(APIConfig):
    def __init__(self):
        super(ArgMinMaxConfig, self).__init__('arg_min_max')
        self.run_torch = False
        self.api_name = 'argmin'
        self.api_list = {
            'argmin': 'argmin',
            'argmax': 'argmax',
        }

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(ArgMinMaxConfig, self).init_from_json(filename, config_id,
                                                    unknown_dim)
        self.feed_spec = [
            {
                "range": [0, 1]
            },  # x
        ]


class PaddleArgMinMax(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)

        result = self.layers(api_name=config.api_name, x=x, axis=-1)

        self.feed_vars = [x]
        self.fetch_vars = [result]


if __name__ == '__main__':
    test_main(pd_dy_obj=PaddleArgMinMax(), config=ArgMinMaxConfig())
