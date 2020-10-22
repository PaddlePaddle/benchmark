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


class MvConfig(APIConfig):
    def __init__(self):
        super(MvConfig, self).__init__("mv")
        self.feed_spec = [{"range": [-1, 1]}, {"range": [-1, 1]}]

    def to_tensorflow(self):
        tf_config = super(MvConfig, self).to_tensorflow()
        tf_config.vec_shape.append(1)
        return tf_config


class PDMv(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        vec = self.variable(
            name='vec', shape=config.vec_shape, dtype=config.vec_dtype)
        result = paddle.mv(x=x, vec=vec)

        self.feed_vars = [x, vec]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x, vec])


class TFMv(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        vec = self.variable(
            name='vec', shape=config.vec_shape, dtype=config.vec_dtype)
        result = tf.matmul(
            a=x,
            b=vec,
            transpose_a=False,
            transpose_b=False,
            adjoint_a=False,
            adjoint_b=False,
            a_is_sparse=False,
            b_is_sparse=False)

        self.feed_list = [x, vec]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, vec])


if __name__ == '__main__':
    test_main(PDMv(), TFMv(), config=MvConfig())
