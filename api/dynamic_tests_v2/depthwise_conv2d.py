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
from conv2d import Conv2dConfig, PDConv2d, TorchConv2d


class DepthwiseConv2dConfig(Conv2dConfig):
    def __init__(self):
        super(DepthwiseConv2dConfig, self).__init__("depthwise_conv2d")

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(DepthwiseConv2dConfig, self).init_from_json(filename, config_id,
                                                          unknown_dim)
        assert self.get_in_channels() == self.groups and self.get_out_channels(
        ) % self.get_in_channels() == 0


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDConv2d(),
        torch_obj=TorchConv2d(),
        config=DepthwiseConv2dConfig())
