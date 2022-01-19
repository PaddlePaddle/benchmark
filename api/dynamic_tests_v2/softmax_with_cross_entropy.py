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


class SoftmaxWithCrossEntropyConfig(APIConfig):
    def __init__(self):
        super(SoftmaxWithCrossEntropyConfig,
              self).__init__("softmax_with_cross_entropy")

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(SoftmaxWithCrossEntropyConfig, self).init_from_json(
            filename, config_id, unknown_dim)
        self.feed_spec = [
            {
                "range": [0, 1]
            },  # input
            {
                "range": [0, self.logits_shape[-1]]
            }  # label
        ]

        logits_rank = len(self.logits_shape)
        if not hasattr(self, "axis") or self.axis == logits_rank - 1:
            self.axis = -1

        if self.soft_label or self.axis != -1:
            print(
                "Warning:\n"
                "  1. PyTorch does not have soft_label param, it only support hard label.\n"
            )
            self.run_torch = False
        else:
            if logits_rank != 2:
                self.logits_shape = [
                    np.prod(self.logits_shape[0:logits_rank - 1]),
                    self.logits_shape[-1]
                ]

            label_rank = len(self.label_shape)
            if label_rank != 2:
                self.label_shape = [
                    np.prod(self.label_shape[0:label_rank - 1]), 1
                ]

    def to_pytorch(self):
        torch_config = super(SoftmaxWithCrossEntropyConfig, self).to_pytorch()
        if self.label_shape[-1] == 1:
            label_rank = len(self.label_shape)
            torch_config.label_shape = [
                np.prod(self.label_shape[0:label_rank - 1])
            ]
        return torch_config


class PDSoftmaxWithCrossEntropy(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        logits = self.variable(
            name="logits",
            shape=config.logits_shape,
            dtype=config.logits_dtype)
        label = self.variable(
            name="label",
            shape=config.label_shape,
            dtype=config.label_dtype,
            stop_gradient=True)
        result = paddle.nn.functional.softmax_with_cross_entropy(
            logits=logits,
            label=label,
            soft_label=config.soft_label,
            ignore_index=config.ignore_index,
            numeric_stable_mode=True,
            return_softmax=False,
            axis=config.axis)

        self.feed_list = [logits, label]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [logits])


class TorchSoftmaxWithCrossEntropy(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name="input", shape=config.logits_shape, dtype=config.logits_dtype)
        label = self.variable(
            name='label',
            shape=config.label_shape,
            dtype=config.label_dtype,
            stop_gradient=True)
        result = torch.nn.functional.cross_entropy(
            input=input,
            target=label,
            weight=None,
            ignore_index=config.ignore_index,
            reduction="none")

        self.feed_list = [input, label]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDSoftmaxWithCrossEntropy(),
        torch_obj=TorchSoftmaxWithCrossEntropy(),
        config=SoftmaxWithCrossEntropyConfig())
