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


class PDCosSim(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.X_shape, dtype=config.X_dtype)
        y = self.variable(name='y', shape=config.Y_shape, dtype=config.Y_dtype)
        result = paddle.fluid.layers.cos_sim(X=x, Y=y)

        self.feed_vars = [x, y]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x, y])


class TFCosSim(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.X_shape, dtype=config.X_dtype)
        y = self.variable(name='y', shape=config.Y_shape, dtype=config.Y_dtype)
        result = tf.compat.v1.losses.cosine_distance(
            labels=x, predictions=y, axis=-1, weights=1.0, scope=None)

        self.feed_list = [x, y]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, y])


if __name__ == '__main__':
    test_main(PDCosSim(), TFCosSim(), config=APIConfig("cos_sim"))
