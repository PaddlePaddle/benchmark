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
        super(GatherConfig, self).__init__("gather")
        self.feed_spec = [
                {"range": [-100, 100] }, #x
                {"range": [0, 1] }  #index
                ]


class PDGather(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(
            name="x", shape=config.x_shape, dtype=config.x_dtype)
        index = self.variable(
            name="index", shape=config.index_shape, dtype='int32')
        result = paddle.gather(x=x, index=index, axis=config.axis)

        self.feed_vars = [x, index]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x, index])


class TFGather(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        params = self.variable(
            name="x", shape=config.x_shape, dtype=config.x_dtype)
        indices = self.variable(
            name="index", shape=config.index_shape, dtype='int32')
        result = tf.gather(params=params, indices=indices, axis=config.axis)

        self.feed_list = [params, indices]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, indices])


if __name__ == '__main__':
    test_main(PDGather(), TFGather(), config=GatherConfig())
