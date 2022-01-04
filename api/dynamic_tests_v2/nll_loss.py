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


class NllLossConfig(APIConfig):
    def __init__(self):
        super(NllLossConfig, self).__init__("nll_loss")

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(NllLossConfig, self).init_from_json(filename, config_id,
                                                  unknown_dim)
        self.feed_spec = [
            {
                "range": [0, 1]
            },  # input
            {
                "range": [0, self.input_shape[-1]]
            }  # label
        ]


class PDNllLoss(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        label = self.variable(
            name='label', shape=config.label_shape, dtype=config.label_dtype)
        result = paddle.nn.functional.nll_loss(
            input=input,
            label=label,
            ignore_index=config.ignore_index,
            reduction='none')

        self.feed_list = [input, label]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


class TorchNllLoss(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        label = self.variable(
            name='label', shape=config.label_shape, dtype=config.label_dtype)
        result = torch.nn.functional.nll_loss(
            input=input,
            target=label,
            ignore_index=config.ignore_index,
            reduction='none')

        self.feed_list = [input, label]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDNllLoss(),
        torch_obj=TorchNllLoss(),
        config=NllLossConfig())
