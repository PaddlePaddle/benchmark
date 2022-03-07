#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class KLDivLossConfig(APIConfig):
    def __init__(self):
        super(KLDivLossConfig, self).__init__("kldiv_loss")
        self.run_torch = False
        self.api_name = 'kldiv_loss'

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(KLDivLossConfig, self).init_from_json(filename, config_id,
                                                    unknown_dim)
        input = np.random.rand(*self.input_shape).astype(self.input_dtype)
        self.input_value = input / input.sum(axis=-1, keepdims=True)

        label = np.random.rand(*self.label_shape).astype(self.label_dtype)
        self.label_value = label / label.sum(axis=-1, keepdims=True)


class PDKLDivLoss(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name='input',
            shape=config.input_shape,
            dtype=config.input_dtype,
            value=config.input_value)
        label = self.variable(
            name='label',
            shape=config.label_shape,
            dtype=config.label_dtype,
            value=config.label_value)
        result = paddle.nn.functional.kl_div(
            input=input, label=label, reduction=config.reduction)

        self.feed_list = [input, label]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


class TorchKLDivLoss(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name='input',
            shape=config.input_shape,
            dtype=config.input_dtype,
            value=config.input_value)
        label = self.variable(
            name='label',
            shape=config.label_shape,
            dtype=config.label_dtype,
            value=config.label_value)
        result = torch.nn.functional.kl_div(
            input=input, target=label, reduction=config.reduction)

        self.feed_list = [input, label]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDKLDivLoss(),
        torch_obj=TorchKLDivLoss(),
        config=KLDivLossConfig())
