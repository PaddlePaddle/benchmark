#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


class ExpandAsConfig(APIConfig):
    def __init__(self):
        super(ExpandAsConfig, self).__init__('expand_as')

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(ExpandAsConfig, self).init_from_json(filename, config_id,
                                                   unknown_dim)
        # if len(self.x_shape) == len(self.shape) + 1:
        #     self.shape = [self.x_shape[0]] + self.shape
        # assert len(self.shape) == len(
        #     self.x_shape), "x and shape should have the same length."


class PDExpandAs(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)
        result = paddle.expand_as(x=x, y=y)

        y.stop_gradient = True
        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TorchExpandAs(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)
        result = x.expand_as(y)

        y.requires_grad = False
        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == "__main__":
    test_main(
        pd_dy_obj=PDExpandAs(),
        torch_obj=TorchExpandAs(),
        config=APIConfig("expand_as"))
