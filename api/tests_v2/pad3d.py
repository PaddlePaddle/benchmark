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


class PDPad3d(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        out = paddle.nn.functional.pad(x,
                                       config.paddings,
                                       mode=config.mode,
                                       value=config.pad_value,
                                       data_format=config.data_format)
        self.feed_vars = [x]
        self.fetch_vars = [out]
        if config.backward:
            self.append_gradients(out, [x])


class TFPad3d(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        padding = [
            [0, 0],
            [0, 0],
            [config.paddings[4], config.paddings[5]],
            [config.paddings[2], config.paddings[3]],
            [config.paddings[0], config.paddings[1]],
        ]
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        out = tf.pad(x,
                     padding,
                     mode=str.upper(config.mode),
                     constant_values=config.pad_value)

        self.feed_list = [x]
        self.fetch_list = [out]
        if config.backward:
            self.append_gradients(out, [x])


if __name__ == '__main__':
    test_main(PDPad3d(), TFPad3d(), config=APIConfig("pad3d"))
