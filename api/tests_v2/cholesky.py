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


class PDCholesky(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        rank = len(config.x_shape)
        x_t = paddle.transpose(x, list(range(rank - 2)) + [rank - 1, rank - 2])
        y = paddle.matmul(x, x_t) + 1e-03
        # x.stop_gradient = False
        result = paddle.cholesky(y, upper=config.upper)
        self.feed_vars = [x]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TFCholesky(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        rank = len(config.x_shape)
        x_t = tf.transpose(x, list(range(rank - 2)) + [rank - 1, rank - 2])
        y = tf.linalg.matmul(x, x_t) + 1e-03
        result = tf.linalg.cholesky(y)
        loss = tf.math.reduce_sum(result)
        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(PDCholesky(), TFCholesky(), config=APIConfig("cholesky"))
