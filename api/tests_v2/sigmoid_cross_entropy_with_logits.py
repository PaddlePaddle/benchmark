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


class PDSigmoidCrossEntropyWithLogits(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        label = self.variable(
            name='label',
            shape=config.label_shape,
            dtype=config.label_dtype,
            stop_gradient=True)
        result = paddle.nn.functional.sigmoid_cross_entropy_with_logits(
            x=x, label=label, ignore_index=config.ignore_index)

        self.feed_vars = [x, label]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TFSigmoidCrossEntropyWithLogits(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        logits = self.variable(
            name='logits', shape=config.x_shape, dtype=config.x_dtype)
        labels = self.variable(
            name='labels', shape=config.label_shape, dtype=config.label_dtype)
        result = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits, labels=labels)

        self.feed_list = [logits, labels]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [logits])


if __name__ == '__main__':
    test_main(
        PDSigmoidCrossEntropyWithLogits(),
        TFSigmoidCrossEntropyWithLogits(),
        config=APIConfig("sigmoid_cross_entropy_with_logits"))
