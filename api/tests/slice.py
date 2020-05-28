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


class SliceConfig(APIConfig):
    def __init__(self):
        super(SliceConfig, self).__init__('slice')
        self.run_tf = False

    def to_tensorflow(self):
        tf_config = super(SliceConfig, self).to_tensorflow()
        if len(self.starts) < len(self.input_shape):
            tf_config.starts = []
            tf_config.ends = []
            for i in range(len(self.input_shape)):
                tf_config.starts.append(0)
                tf_config.ends.append(self.input_shape[i])
            for i in self.axes:
                tf_config.starts[i] = self.starts[i]
                tf_config.ends[i] = self.ends[i] - self.starts[i]
        else:
            for j in range(len(self.starts)):
                tf_config.ends[j] = self.ends[j] - self.starts[j]
        return tf_config


class PDSlice(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        input = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        result = fluid.layers.slice(
            input=input,
            axes=config.axes,
            starts=config.starts,
            ends=config.ends)

        self.feed_vars = [input]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [input])


class TFSlice(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        begin = self.variable(name='begin', shape=config.starts, dtype="int32")
        size = self.variable(name='size', shape=config.ends, dtype="int32")
        result = tf.slice(input_=input, begin=begin, size=size)

        self.feed_list = [input, begin, size]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input, begin, size])


if __name__ == '__main__':
    test_main(PDSlice(), TFSlice(), config=SliceConfig())
