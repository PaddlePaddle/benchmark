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


class InstanceNormConfig(APIConfig):
    def __init__(self):
        super(InstanceNormConfig, self).__init__('instance_norm')
        self.run_tf = False


class PDInstanceNorm(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='input',
                shape=config.input_shape,
                dtype=config.input_dtype,
                lod_level=0)
            data.stop_gradient = False
            result = fluid.layers.instance_norm(
                input=data, epsilon=config.epsilon)

            self.feed_vars = [data]
            self.fetch_vars = [result]


class TFInstanceNorm(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        data = self.placeholder(
            name='input', shape=config.input_shape, dtype=config.input_dtype)


def register_api():
    REGISTER_API_INFO[
        'instance_norm'] = ['instance_norm', 'instance_norm.json']


if __name__ == '__main__':
    test_main(PDInstanceNorm(), TFInstanceNorm(), config=InstanceNormConfig())
