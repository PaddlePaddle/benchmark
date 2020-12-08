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


class LayerNormConfig(APIConfig):
    def __init__(self):
        super(LayerNormConfig, self).__init__("layer_norm")


class PaddleLayerNorm(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name="x", shape=config.x_shape, dtype=config.x_dtype)
        self.feed_list = [x]

    def run_graph(self, config):
        x = self.feed_list[0]
        layer_norm = paddle.nn.LayerNorm(x.shape[1:])
        result = layer_norm(self.feed_list[0])
        # result = paddle.nn.functional.layer_norm(
        #     x=self.feed_list[0], normalized_shape=config.x_shape[1:], epsilon=config.epsilon)
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, self.feed_list)


class TorchLayerNorm(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        self.feed_list = [x]

    def run_graph(self, config):
        x = self.feed_list[0]
        # layer_norm = torch.nn.LayerNorm(x.shape[1:])
        # result = layer_norm(self.feed_list[0])
        result = torch.nn.functional.layer_norm(x, x.shape[1:])
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, self.feed_list)


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PaddleLayerNorm(),
        torch_obj=TorchLayerNorm(),
        config=LayerNormConfig())
