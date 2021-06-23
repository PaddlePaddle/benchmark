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
from reduce import PDReduce, TorchReduce


class ReduceAllAnyConfig(APIConfig):
    def __init__(self):
        super(ReduceAllAnyConfig, self).__init__('reduce_all_any')
        self.feed_spec = {"range": [-1, 1]}
        self.api_name = 'all'
        self.api_list = {'all': 'all', 'any': 'any'}

    def init_from_json(self, filename, config_id=3, unknown_dim=16):
        super(ReduceAllAnyConfig, self).init_from_json(filename, config_id,
                                                       unknown_dim)
        if self.axis == None:
            self.axis = []


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDReduce(),
        torch_obj=TorchReduce(),
        config=ReduceAllAnyConfig())
