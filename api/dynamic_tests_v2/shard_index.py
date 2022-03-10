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


class ShardIndexConfig(APIConfig):
    def __init__(self):
        super(ShardIndexConfig, self).__init__("shard_index")
        self.feed_spec = {"range": [0, 99999]}
        self.run_torch = False


class PDShardIndex(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name="input", shape=config.input_shape, dtype=config.input_dtype)
        result = paddle.shard_index(input=input, index_num=100000, nshards=100, shard_id=0)
        self.feed_list = [input]
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(pd_dy_obj=PDShardIndex(), config=ShardIndexConfig())
