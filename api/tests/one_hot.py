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


class OneHotConfig(APIConfig):
    def __init__(self):
        super(OneHotConfig, self).__init__('one_hot')

    def init_from_json(self, filename, config_id=0):
        super(OneHotConfig, self).init_from_json(filename, config_id)
        self.feed_spec = {"range": [0, self.depth]}


class PDOneHot(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='data',
                shape=config.input_shape,
                dtype=config.input_dtype,
                lod_level=0)
            data.stop_gradient = False
            result = fluid.one_hot(input=data, depth=config.depth)

            self.feed_vars = [data]
            self.fetch_vars = [result]
            if config.backward:
                self.append_gradients(result, [data])


class TFOneHot(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        data = self.placeholder(
            name='data', shape=config.input_shape, dtype=config.input_dtype)
        result = tf.one_hot(
            indices=data,
            depth=config.depth,
            on_value=None,
            off_value=None,
            axis=None,
            dtype=None)

        self.feed_list = [data]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [data])


def register_api():
    REGISTER_API_INFO['one_hot'] = ['one_hot', 'one_hot.json']


if __name__ == '__main__':
    test_main(PDOneHot(), TFOneHot(), config=OneHotConfig())
