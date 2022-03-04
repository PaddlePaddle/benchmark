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


class SegmentPoolConfig(APIConfig):
    def __init__(self):
        super(SegmentPoolConfig, self).__init__('segment_pool')
        self.run_torch = False

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(SegmentPoolConfig, self).init_from_json(filename, config_id,
                                                      unknown_dim)
        self.feed_spec = [
            {
                "range": [0, 1]
            },  # x
        ]

        segment = np.random.randint(
            0, self.x_shape[0] // 20, size=[self.x_shape[0]])
        segment = np.sort(segment)
        self.segment_ids_value = segment.astype(self.segment_ids_dtype)


class PaddleSegmentPool(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        segment_ids = self.variable(
            name='segment_ids',
            shape=config.segment_ids_shape,
            dtype=config.segment_ids_dtype,
            value=config.segment_ids_value)
        result = getattr(paddle.incubate, config.pool_type)(x, segment_ids)

        self.feed_vars = [x]
        self.fetch_vars = [result]


if __name__ == '__main__':
    test_main(pd_dy_obj=PaddleSegmentPool(), config=SegmentPoolConfig())
