#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from elementwise import ElementwiseConfig


@benchmark_registry.register("divide", reuse="elementwise")
class DivideConfig(ElementwiseConfig):
    def __init__(self):
        super(DivideConfig, self).__init__('divide')
        self.feed_spec = [{"range": [1, 3]}, {"range": [1, 3]}]
        self.alias_name = 'elementwise'
        self.api_name = 'divide'
        self.api_list = {'divide': 'divide'}

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(DivideConfig, self).init_from_json(filename, config_id,
                                                 unknown_dim)
        if self.x_dtype == 'float32':
            self.atol = 1e-4
        else:
            self.atol = 0.5
