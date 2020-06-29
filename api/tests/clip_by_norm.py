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


class ClipByNormConfig(APIConfig):
    def __init__(self):
        super(ClipByNormConfig, self).__init__("clip_by_norm")
        self.feed_spec = {"range": [-10, 10]}


class PDClipByNorm(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name="x", shape=config.x_shape, dtype=config.x_dtype)
        result = fluid.layers.clip_by_norm(x=x, max_norm=2.0)

        self.feed_vars = [x]
        self.fetch_vars = [result]


class TFClipByNorm(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = tf.clip_by_norm(t=x, clip_norm=2.0, axes=None)

        self.feed_list = [x]
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(PDClipByNorm(), TFClipByNorm(), config=ClipByNormConfig())
