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


class SplitConfig(APIConfig):
    def __init__(self):
        super(SplitConfig, self).__init__('split')

    def to_pytorch(self):
        torch_config = super(SplitConfig, self).to_pytorch()
        torch_config.num_or_sections = int(self.x_shape[self.axis] / self.num_or_sections)
        return torch_config


class PDSplit(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = paddle.split(
            x=x, num_or_sections=config.num_or_sections, axis=config.axis)
        if type(result) == list:
            self.fetch_list = result
            # computing a list of gradients is not supported
            config.backward = False
        else:
            self.fetch_list = [result]
        self.feed_list = [x]
        if config.backward:
            self.append_gradients(result, [x])


class TorchSplit(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = torch.split(
            tensor=x, split_size_or_sections=config.num_or_sections, dim=config.axis)

        if type(result) == tuple:
            result = list(result)
            self.fetch_list = result
            # computing a list of gradients is not supported
            config.backward = False
        else:
            self.fetch_list = [result]
        self.feed_list = [x]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDSplit(), torch_obj=TorchSplit(), config=SplitConfig())
