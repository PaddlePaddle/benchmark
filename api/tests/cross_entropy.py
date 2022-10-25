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


@benchmark_registry.register("cross_entropy")
class CrossEntropyConfig(APIConfig):
    def __init__(self):
        super(CrossEntropyConfig, self).__init__("cross_entropy")

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(CrossEntropyConfig, self).init_from_json(filename, config_id,
                                                       unknown_dim)
        input_rank = len(self.input_shape)
        if not hasattr(self, "axis") or self.axis == input_rank - 1:
            self.axis = -1

        self.num_classes = self.input_shape[self.axis]
        self.feed_spec = [
            {
                "range": [0, 1]
            },  # input
            {
                "range": [0, self.num_classes]
            }  # label
        ]

        if self.label_dtype in ['float32', 'float64'] or self.axis != -1:
            self.run_tf = False

        if self.soft_label or self.axis != -1:
            print(
                "Warning:\n"
                "  1. PyTorch does not have soft_label param, it only support hard label.\n"
            )
            self.run_torch = False
        else:
            if input_rank != 2:
                self.input_shape = [
                    np.prod(self.input_shape[0:input_rank - 1]),
                    self.input_shape[-1]
                ]

            label_rank = len(self.label_shape)
            if label_rank != 2:
                self.label_shape = [
                    np.prod(self.label_shape[0:label_rank - 1]), 1
                ]

    def to_pytorch(self):
        torch_config = super(CrossEntropyConfig, self).to_pytorch()
        if self.label_shape[-1] == 1:
            label_rank = len(self.label_shape)
            torch_config.label_shape = [
                np.prod(self.label_shape[0:label_rank - 1])
            ]
        return torch_config

    def to_tensorflow(self):
        tf_config = super(CrossEntropyConfig, self).to_tensorflow()
        label_rank = len(tf_config.label_shape)
        if tf_config.label_shape[label_rank - 1] == 1:
            tf_config.label_shape = tf_config.label_shape[0:label_rank - 1]
        return tf_config


@benchmark_registry.register("cross_entropy")
class PaddleCrossEntropy(PaddleOpBenchmarkBase):
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
            soft_label=config.soft_label,
            use_softmax=True,
            axis=config.axis,
            reduction="none")

        self.feed_list = [input, label]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


@benchmark_registry.register("cross_entropy")
class TorchCrossEntropy(PytorchOpBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name="input", shape=config.input_shape, dtype=config.input_dtype)
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


@benchmark_registry.register("cross_entropy")
class TFCrossEntropy(TensorflowOpBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        label = self.variable(
            name='label', shape=config.label_shape, dtype=config.label_dtype)
        onehot_label = tf.one_hot(indices=label, depth=config.num_classes)
        result = tf.compat.v1.losses.softmax_cross_entropy(
            input=input, onehot_labels=onehot_label, reduction='none')

        self.feed_list = [input, label]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])
