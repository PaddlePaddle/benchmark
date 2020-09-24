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


class GatherConfig(APIConfig):
    def __init__(self):
        super(GatherConfig, self).__init__('gather')

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(GatherConfig, self).init_from_json(filename, config_id,
                                                 unknown_dim)
        self.feed_spec = [
            {
                "range": [8, 10]
            },  # input
            {
                "range": [1, self.input_shape[self.axis]]
            }  # index
        ]


class PDGather(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(
            name='x', shape=config.input_shape, dtype=config.input_dtype)
        index = self.variable(
            name='index',
            shape=config.index_shape,
            dtype=config.index_dtype,
            stop_gradient=True)
        result = paddle.gather(x=x, index=index, axis=config.axis)

        self.feed_vars = [x, index]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TFGather(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        params = self.variable(
            name='params', shape=config.input_shape, dtype=config.input_dtype)
        indices = self.variable(
            name='indices', shape=config.index_shape, dtype=config.index_dtype)
        result = tf.gather(params=params, indices=indices, axis=config.axis)

        self.feed_list = [params, indices]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [params])


if __name__ == '__main__':
    test_main(PDGather(), TFGather(), config=GatherConfig())
