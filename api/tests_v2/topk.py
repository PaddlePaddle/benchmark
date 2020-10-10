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


class PDTopK(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        data = self.variable(
            name='x', shape=config.x_shape, dtype=config.x_dtype)
        value, indices = paddle.topk(x=data, k=config.k)

        self.feed_vars = [data]
        self.fetch_vars = [value, indices]
        if config.backward:
            self.append_gradients([value], [data])
        print(self.fetch_vars)


class TFTopK(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        data = self.variable(
            name='x', shape=config.x_shape, dtype=config.x_dtype)
        value, indices = tf.math.top_k(input=data, k=config.k)

        self.feed_list = [data]
        self.fetch_list = [value, indices]
        if config.backward:
            self.append_gradients([value], [data])
        print(self.fetch_list)


if __name__ == '__main__':
    test_main(PDTopK(), TFTopK(), config=APIConfig("topk"))
