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


@benchmark_registry.register("graph_send_recv")
class GraphSendRecvConfig(APIConfig):
    def __init__(self):
        super(GraphSendRecvConfig, self).__init__('graph_send_recv')

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(GraphSendRecvConfig, self).init_from_json(filename, config_id,
                                                        unknown_dim)
        self.feed_spec = [
            {
                "range": [0, 1]
            },  # input
            {
                "range": [1, self.x_shape[0]]
            },  # src_index
            {
                "range": [1, self.x_shape[0]]
            }  # dst_index
        ]


@benchmark_registry.register("graph_send_recv")
class PaddleGraphSendRecv(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        src_index = self.variable(
            name='src_index',
            shape=config.src_index_shape,
            dtype=config.src_index_dtype)
        dst_index = self.variable(
            name='dst_index',
            shape=config.dst_index_shape,
            dtype=config.dst_index_dtype)
        result = paddle.incubate.graph_send_recv(
            x, src_index, dst_index, pool_type=config.pool_type)

        self.feed_vars = [x, src_index, dst_index]
        self.fetch_vars = [result]
