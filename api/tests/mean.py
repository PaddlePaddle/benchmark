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


class MeanConfig(APIConfig):
    def __init__(self):
        super(MeanConfig, self).__init__('mean')
        self.run_tf = False


class PDMean(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='x',
                shape=config.x_shape,
                dtype=config.x_dtype,
                lod_level=0)
            data.stop_gradient = False
            result = fluid.layers.mean(x=data)

            self.feed_vars = [data]
            self.fetch_vars = [result]


class TFMean(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        data = self.placeholder(
            name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = tf.keras.metrics.Mean()

        self.feed_list = [data]
        self.fetch_list = [result]


def register_api():
    REGISTER_API_INFO['mean'] = ['mean', 'mean.json']


if __name__ == '__main__':
    test_main(PDMean(), TFMean(), config=MeanConfig())
