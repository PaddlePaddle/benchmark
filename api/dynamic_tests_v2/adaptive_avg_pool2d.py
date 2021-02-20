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

class AdaptiveAvgPool2dConfig(APIConfig):
    def __init__(self):
        super(AdaptiveAvgPool2dConfig, self).__init__("adaptive_avg_pool2d")

class PaddleAdaptiveAvgPool2D(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        adaptive_avg_pool2d = paddle.nn.AdaptiveAvgPool2D(
            output_size=config.output_size,
            data_format=config.data_format)
        result = adaptive_avg_pool2d(x)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])

class TorchAdaptiveAvgPool2D(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name="x", shape=config.x_shape, dtype=config.x_dtype)
        adaptiveavgpool2d = torch.nn.AdaptiveAvgPool2d(output_size=config.output_size)
        result = adaptiveavgpool2d(x)
        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PaddleAdaptiveAvgPool2D(),
        torch_obj=TorchAdaptiveAvgPool2D(),
        config=AdaptiveAvgPool2dConfig())
