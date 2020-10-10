#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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


class LayerNormConfig(APIConfig):
    def __init__(self):
        super(LayerNormConfig, self).__init__('layer_norm')
        self.run_tf = False


class PDLayerNorm(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(
            name='data', shape=config.x_shape, dtype=config.x_dtype)
        result = paddle.nn.functional.layer_norm(
            x=x, normalized_shape=config.x_shape[1:], epsilon=config.epsilon)

        self.feed_vars = [x]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TFLayerNorm(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        data = self.variable(
            name='data', shape=config.x_shape, dtype=config.x_dtype)


if __name__ == '__main__':
    test_main(PDLayerNorm(), TFLayerNorm(), config=LayerNormConfig())
