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


class PDSigmoidCrossEntropyWithLogits(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            x = fluid.data(
                name='x',
                shape=config.x_shape,
                dtype=config.x_dtype,
                lod_level=0)
            label = fluid.data(
                name='label',
                shape=config.label_shape,
                dtype=config.label_dtype,
                lod_level=0)
            x.stop_gradient = False
            label.stop_gradient = False
            result = fluid.layers.sigmoid_cross_entropy_with_logits(
                x=x, label=label, ignore_index=config.ignore_index)

            self.feed_vars = [x, label]
            self.fetch_vars = [result]
            if config.backward:
                self.append_gradients(result, [x, label])


class TFSigmoidCrossEntropyWithLogits(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        logits = self.placeholder(
            name='logits', shape=config.x_shape, dtype=config.x_dtype)
        labels = self.placeholder(
            name='labels', shape=config.label_shape, dtype=config.label_dtype)
        result = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=labels, labels=logits)

        self.feed_list = [labels, logits]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [labels, logits])


def register_api():
    REGISTER_API_INFO['sigmoid_cross_entropy_with_logits'] = [
        'sigmoid_cross_entropy_with_logits',
        'sigmoid_cross_entropy_with_logits.json'
    ]


if __name__ == '__main__':
    test_main(
        PDSigmoidCrossEntropyWithLogits(),
        TFSigmoidCrossEntropyWithLogits(),
        config=APIConfig("sigmoid_cross_entropy_with_logits"))
