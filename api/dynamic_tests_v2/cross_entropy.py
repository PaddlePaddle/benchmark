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


class CrossEntropyConfig(APIConfig):
    def __init__(self):
        super(CrossEntropyConfig, self).__init__("cross_entropy")
        self.alias_name = "softmax_with_cross_entropy"

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(CrossEntropyConfig, self).init_from_json(filename, config_id,
                                                       unknown_dim)
        self.feed_spec = [
            {
                "range": [0, 1]
            },  # input
            {
                "range": [0, self.logits_shape[-1]]
            }  # label
        ]

        if self.soft_label:
            print(
                "Warning:\n"
                "  1. PyTorch does not have soft_label param, it only support hard label.\n"
            )
            self.run_torch = False

        if not hasattr(self, "reduction"):
            self.reduction = "none"
        if not hasattr(self, "axis"):
            self.axis = -1

    def to_pytorch(self):
        torch_config = super(CrossEntropyConfig, self).to_pytorch()
        logits_rank = len(self.logits_shape)
        if logits_rank != 2:
            torch_config.logits_shape = [
                np.prod(self.logits_shape[0:logits_rank - 1]),
                self.logits_shape[-1]
            ]
        if self.label_shape[-1] == 1:
            label_rank = len(self.label_shape)
            torch_config.label_shape = [
                np.prod(self.label_shape[0:label_rank - 1])
            ]
        return torch_config


class PDCrossEntropy(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name="input", shape=config.logits_shape, dtype=config.logits_dtype)
        label = self.variable(
            name="label",
            shape=config.label_shape,
            dtype=config.label_dtype,
            stop_gradient=True)
        result = paddle.nn.functional.cross_entropy(
            input=input,
            label=label,
            weight=None,
            ignore_index=config.ignore_index,
            reduction=config.reduction,
            soft_label=config.soft_label,
            axis=config.axis)

        self.feed_list = [input, label]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


class TorchCrossEntropy(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name="input", shape=config.logits_shape, dtype=config.logits_dtype)
        label = self.variable(
            name='label', shape=config.label_shape, dtype=config.label_dtype)
        result = torch.nn.functional.cross_entropy(
            input=input,
            target=label,
            weight=None,
            size_average=None,
            ignore_index=config.ignore_index,
            reduction=config.reduction)

        self.feed_list = [input, label]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDCrossEntropy(),
        torch_obj=TorchCrossEntropy(),
        config=CrossEntropyConfig())
