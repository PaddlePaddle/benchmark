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
        tf_config = self
        tf_config.ends = fluid.layers.elementwise_sub(self.starts, self.ends)
        return tf_config


class PDSlice(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='input',
                shape=config.input_shape,
                dtype=config.input_dtype,
                lod_level=0)
            data.stop_gradient = False
            result = fluid.layers.slice(
                input=data,
                axes=config.axes,
                starts=config.starts,
                ends=config.ends)

            self.feed_vars = [data]
            self.fetch_vars = [result]
            if config.backward:
                self.append_gradients(result, [data])


class TFSlice(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        data = self.placeholder(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        result = tf.slice(input_=data, begin=config.starts, size=config.ends)

        self.feed_list = [data]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [data])


if __name__ == '__main__':
    test_main(PDSlice(), TFSlice(), config=APIConfig("slice"))
