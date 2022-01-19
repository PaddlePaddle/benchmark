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
        data = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        result = fluid.layers.instance_norm(input=data, epsilon=config.epsilon)

        self.feed_vars = [data]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [data])


class TFInstanceNorm(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        data = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)


if __name__ == '__main__':
    test_main(PDInstanceNorm(), TFInstanceNorm(), config=InstanceNormConfig())
