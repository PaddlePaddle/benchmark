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

from main import test_main

import sys
sys.path.append("..")
from common import paddle_api_benchmark as paddle_api
from common import tensorflow_api_benchmark as tensorflow_api


class SoftmaxWithCrossEntropyConfig(object):
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.num_classes = input_shape[1]
        self.feed_spec = [
            {
                "range": [0, 1]
            },  # input
            {
                "range": [0, self.num_classes]
            }  # label
        ]

    def label_shape(self, for_tensorflow=False):
        if not for_tensorflow:
            return [self.input_shape[0], 1]
        else:
            return [self.input_shape[0]]


config = SoftmaxWithCrossEntropyConfig(input_shape=[128, 100])


class PDSoftmaxWithCrossEntropy(paddle_api.PaddleAPIBenchmarkBase):
    def build_program(self, backward=False, dtype=None):
        import paddle.fluid as fluid

        self.name = "softmax_with_cross_entropy"
        with fluid.program_guard(self.main_program, self.startup_program):
            input = fluid.data(
                name='input',
                shape=config.input_shape,
                dtype='float32',
                lod_level=0)
            label = fluid.data(
                name="label",
                shape=config.label_shape(),
                dtype="int64",
                lod_level=0)
            input.stop_gradient = False
            result = fluid.layers.softmax_with_cross_entropy(
                logits=input, label=label, soft_label=False)

            self.feed_vars = [input, label]
            self.fetch_vars = [result]
            if backward:
                self.append_gradients(result, [input])


class TFSoftmaxWithCrossEntropy(tensorflow_api.TensorflowAPIBenchmarkBase):
    def build_graph(self, backward=False, dtype=None):
        import tensorflow as tf

        self.name = "softmax_with_cross_entropy"
        self.allow_growth = True

        input = tf.placeholder(
            name='input', shape=config.input_shape, dtype=tf.float32)
        label = tf.placeholder(
            name='label',
            shape=config.label_shape(for_tensorflow=True),
            dtype=tf.int32)
        onehot_label = tf.one_hot(indices=label, depth=config.num_classes)
        result = tf.losses.softmax_cross_entropy(
            logits=input, onehot_labels=onehot_label)

        self.feed_list = [input, label]
        self.fetch_list = [result]
        if backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    # Not consistent!!!
    test_main(
        PDSoftmaxWithCrossEntropy(),
        TFSoftmaxWithCrossEntropy(),
        feed_spec=config.feed_spec)
