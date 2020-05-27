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
        self.run_tf = False

    def init_from_json(self, filename, config_id=0):
        super(SoftmaxWithCrossEntropyConfig, self).init_from_json(filename,
                                                                  config_id)
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

    def to_tensorflow(self):
        tf_config = super(FCConfig, self).to_tensorflow()
        label_rank = len(tf_config.label_shape)
        tf_config.label_shape = tf_config.label_shape[0:label_rank - 2]
        return tf_config


class PDSoftmaxWithCrossEntropy(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        input = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        label = self.variable(
            name="label",
            shape=config.label_shape,
            dtype=config.label_dtype,
            stop_gradient=True)
        result = fluid.layers.softmax_with_cross_entropy(
            logits=input, label=label, soft_label=config.soft_label)

        self.feed_vars = [input, label]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [input])


class TFSoftmaxWithCrossEntropy(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        label = self.variable(
            name='label', shape=config.label_shape, dtype=config.label_dtype)
        onehot_label = tf.one_hot(indices=label, depth=config.num_classes)
        result = tf.losses.softmax_cross_entropy(
            logits=input, onehot_labels=onehot_label)

        self.feed_list = [input, label]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    test_main(
        PDSoftmaxWithCrossEntropy(),
        TFSoftmaxWithCrossEntropy(),
        config=SoftmaxWithCrossEntropyConfig())
