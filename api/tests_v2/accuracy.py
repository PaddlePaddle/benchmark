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


class AccuracyConfig(APIConfig):
    def __init__(self):
        super(AccuracyConfig, self).__init__('accuracy')

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(AccuracyConfig, self).init_from_json(filename, config_id,
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

    def to_tensorflow(self):
        tf_config = super(AccuracyConfig, self).to_tensorflow()
        label_rank = len(tf_config.label_shape)
        if tf_config.label_shape[label_rank - 1] == 1:
            tf_config.label_shape = tf_config.label_shape[0:label_rank - 1]
        return tf_config


class PDAccuracy(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        input = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        label = self.variable(
            name='label', shape=config.label_shape, dtype=config.label_dtype)
        result = paddle.metric.accuracy(
            input=input, label=label, k=1, correct=None, total=None)

        self.feed_vars = [input, label]
        self.fetch_vars = [result[0]]


class TFAccuracy(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        predictions = self.variable(
            name='predictions',
            shape=config.input_shape,
            dtype=config.input_dtype)
        labels = self.variable(
            name='labels', shape=config.label_shape, dtype=config.label_dtype)
        predictions_argmax = tf.compat.v1.argmax(predictions,
                                                 len(config.input_shape) - 1)
        _, result = tf.compat.v1.metrics.accuracy(
            labels=labels, predictions=predictions_argmax)

        self.feed_list = [predictions, labels]
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(PDAccuracy(), TFAccuracy(), config=AccuracyConfig())
