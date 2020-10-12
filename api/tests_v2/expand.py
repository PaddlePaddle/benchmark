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


class ExpandConfig(APIConfig):
    def __init__(self):
        super(ExpandConfig, self).__init__('expand')

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(ExpandConfig, self).init_from_json(filename, config_id,
                                                 unknown_dim)
        if len(self.x_shape) == len(self.shape) + 1:
            self.shape = [self.x_shape[0]] + self.shape
        assert len(self.x_shape) == len(
            self.shape
        ), "The length of shape should be equal to the rank of input x."

    def to_tensorflow(self):
        tf_config = super(ExpandConfig, self).to_tensorflow()
        tf_config.multiples = []
        for i in range(len(self.x_shape)):
            tf_config.multiples.append(self.shape[i] // self.x_shape[i])
        return tf_config


class PDExpand(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = paddle.expand(x=x, shape=config.shape)

        self.feed_vars = [x]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TFExpand(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = tf.tile(input=x, multiples=config.multiples)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(PDExpand(), TFExpand(), config=ExpandConfig())
