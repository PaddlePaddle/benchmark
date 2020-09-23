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
        a_var = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)#, value=a)
        # print("paddle input", a_var.data)
        rank = len(config.x_shape)
        a_t = paddle.transpose(a_var, list(range(rank-2)) + [rank-1, rank-2])
        x = paddle.matmul(a_var, a_t) + 1e-03
        x.stop_gradient = False
        l = paddle.cholesky(x, upper=config.upper)
        self.feed_vars = [a_var]
        self.fetch_vars = [l]
        # if config.backward:
        #     self.append_gradients(l, [x])


class TFCholesky(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        a_var = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)#, value=inp)
        # print("tf input", a_var)
        rank = len(config.x_shape)
        a_t = tf.transpose(a_var, list(range(rank-2)) + [rank-1, rank-2])
        x = tf.linalg.matmul(a_var, a_t) + 1e-03
        l = tf.linalg.cholesky(x)
        loss = tf.math.reduce_sum(l)
        self.feed_list = [a_var]
        self.fetch_list = [l]
        # if config.backward:
        #     self.append_gradients(l, [x])


if __name__ == '__main__':
     test_main(PDCholesky(), TFCholesky(), config=APIConfig("cholesky"))