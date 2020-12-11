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

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(CrossEntropyConfig, self).init_from_json(filename, config_id,
                                                       unknown_dim)
        input_rank = len(self.input_shape)
        self.num_classes = self.input_shape[input_rank - 1]
        self.feed_spec = [
            {
                "range": [0, 1]
            },  # input
            {
                "range": [0, self.num_classes]
            }  # label
        ]


class PDCrossEntropy(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name="input", shape=config.input_shape, dtype=config.input_dtype)
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
            name="input", shape=config.input_shape, dtype=config.input_dtype)
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
