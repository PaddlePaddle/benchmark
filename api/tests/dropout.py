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


class PDDropout(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            x = fluid.data(
                name='x',
                shape=config.x_shape,
                dtype=config.x_dtype,
                lod_level=0)
            x.stop_gradient = False
            result = fluid.layers.dropout(
                x=x,
                dropout_prob=config.dropout_prob,
                seed=123,
                dropout_implementation=config.dropout_implementation)

            self.feed_vars = [x]
            self.fetch_vars = [result]
            if config.backward:
                self.append_gradients(result, [x])


class TFDropout(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.placeholder(
            name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = tf.nn.dropout(
            x=x, rate=config.dropout_prob, noise_shape=None, seed=123)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


def register_api():
    REGISTER_API_INFO['dropout'] = ['dropout', 'dropout.json']


if __name__ == '__main__':
    test_main(PDDropout(), TFDropout(), config=APIConfig("dropout"))
