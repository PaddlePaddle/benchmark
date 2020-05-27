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


#TODO:parameter total is not include
class AccuracyConfig(APIConfig):
    def __init__(self):
        super(AccuracyConfig, self).__init__('accuracy')
        self.run_tf = False

    def to_tensorflow(self):
        tf_config = super(AccuracyConfig, self).to_tensorflow()
        tf_config.label_shape = self.input_shape
        return tf_config


class PDAccuracy(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        input = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        label = self.variable(
            name='label', shape=config.label_shape, dtype=config.label_dtype)
        result = fluid.layers.accuracy(input=input, label=label, k=1)

        self.feed_vars = [input, label]
        self.fetch_vars = [result[0]]
        if config.backward:
            self.append_gradients(result[0], [input, label])


# The labels of accuracy in Paddle and TF is not same. 
class TFAccuracy(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        predictions = self.variable(
            name='predictions',
            shape=config.input_shape,
            dtype=config.input_dtype)
        labels = self.variable(
            name='labels', shape=config.label_shape, dtype=config.label_dtype)
        result1, result2 = tf.compat.v1.metrics.accuracy(
            labels=labels, predictions=predictions)

        self.feed_list = [predictions, labels]
        self.fetch_list = [result2]
        if config.backward:
            self.append_gradients(result2, [predictions, labels])


if __name__ == '__main__':
    test_main(PDAccuracy(), TFAccuracy(), config=AccuracyConfig())
