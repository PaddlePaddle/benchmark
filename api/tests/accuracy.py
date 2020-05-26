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

#        self.run_tf = False

    def to_tensorflow(self):
        tf_config = self
        tf_config.label_shape = self.input_shape
        return tf_config


class PDAccuracy(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            input = fluid.data(
                name='input',
                shape=config.input_shape,
                dtype=config.input_dtype,
                lod_level=0)
            label = fluid.data(
                name='label',
                shape=config.label_shape,
                dtype=config.label_dtype,
                lod_level=0)
            input.stop_gradient = False
            label.stop_gradient = False
            result = fluid.layers.accuracy(input=input, label=label, k=1)

            self.feed_vars = [input, label]
            self.fetch_vars = [result[0]]
            if config.backward:
                self.append_gradients(result[0], [input, label])


class TFAccuracy(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        predictions = self.placeholder(
            name='predictions',
            shape=config.input_shape,
            dtype=config.input_dtype)
        labels = self.placeholder(
            name='labels', shape=config.label_shape, dtype=config.label_dtype)
        result1, result2 = tf.compat.v1.metrics.accuracy(
            labels=labels, predictions=predictions)

        self.feed_list = [predictions, labels]
        self.fetch_list = [result2]
        if config.backward:
            self.append_gradients(result2, [predictions, labels])


if __name__ == '__main__':
    test_main(PDAccuracy(), TFAccuracy(), config=AccuracyConfig())
