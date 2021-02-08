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


class Pad3dConfig(APIConfig):
    def __init__(self):
        super(Pad3dConfig, self).__init__('pad3d')

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(Pad3dConfig, self).init_from_json(filename, config_id,
                                                unknown_dim)
        # in API 'torch.nn.functional.pad()', 'reflect' mode is not implemented,
        # while in API 'torch.nn.ReplicationPad3d()', the logic of 'reflection' 
        # mode is different from paddle.pad(mode='reflect')
        # So set run_torch as False when using 'reflect' mode
        if self.mode == "reflect":
            print(
                "Warning:\n"
                "  1. 'reflect' mode in torch.nn.functional.pad is not implemented.\n"
            )
            self.run_torch = False


class PaddlePad3d(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = paddle.nn.functional.pad(x=x,
                                          pad=config.pad,
                                          mode=config.mode,
                                          data_format=config.data_format)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TorchPad3d(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = torch.nn.functional.pad(input=x,
                                         pad=config.pad,
                                         mode=config.mode,
                                         value=config.value)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PaddlePad3d(), torch_obj=TorchPad3d(), config=Pad3dConfig())
