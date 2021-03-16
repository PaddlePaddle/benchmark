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


class BinaryCrossEntropyWithLogitsConfig(APIConfig):
    def __init__(self):
        super(BinaryCrossEntropyWithLogitsConfig,
              self).__init__("binary_cross_entropy_with_logits")

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(BinaryCrossEntropyWithLogitsConfig, self).init_from_json(
            filename, config_id, unknown_dim)
        self.feed_spec = [
            {
                "range": [0, 1]
            },  # logit
            {
                "range": [0, self.logit_shape[-1]]
            }  # label
        ]


class PaddleBinaryCrossEntropyWithLogits(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        logit = self.variable(
            name="logit", shape=config.logit_shape, dtype=config.logit_dtype)
        label = self.variable(
            name="label",
            shape=config.label_shape,
            dtype=config.label_dtype,
            stop_gradient=True)
        result = paddle.nn.functional.binary_cross_entropy_with_logits(
            logit=logit,
            label=label,
            weight=None,
            reduction="none",
            pos_weight=None)

        self.feed_list = [logit, label]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [logit])


class TorchBinaryCrossEntropyWithLogits(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        logit = self.variable(
            name="logit", shape=config.logit_shape, dtype=config.logit_dtype)
        label = self.variable(
            name="label",
            shape=config.label_shape,
            dtype=config.label_dtype,
            stop_gradient=True)
        result = torch.nn.functional.binary_cross_entropy_with_logits(
            input=logit, target=label, weight=None, reduction="none")

        self.feed_list = [logit]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [logit])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PaddleBinaryCrossEntropyWithLogits(),
        torch_obj=TorchBinaryCrossEntropyWithLogits(),
        config=BinaryCrossEntropyWithLogitsConfig())
