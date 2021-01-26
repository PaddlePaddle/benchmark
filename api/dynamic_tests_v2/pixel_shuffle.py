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


class PixelShuffleConfig(APIConfig):
    def __init__(self):
        super(PixelShuffleConfig, self).__init__('pixel_shuffle')

    def init_from_json(self, filename, config_id=1, unknown_dim=16):
        super(PixelShuffleConfig, self).init_from_json(filename, config_id,
                                                       unknown_dim)
        if self.data_format == 'NHWC':
            self.run_torch = False


class PDPixelShuffle(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = paddle.nn.functional.pixel_shuffle(
            x=x,
            upscale_factor=config.upscale_factor,
            data_format=config.data_format)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TorchPixelShuffle(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = torch.nn.functional.pixel_shuffle(
            input=x, upscale_factor=config.upscale_factor)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDPixelShuffle(),
        torch_obj=TorchPixelShuffle(),
        config=PixelShuffleConfig())
