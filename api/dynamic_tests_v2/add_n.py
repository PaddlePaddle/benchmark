#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


class PDAddN(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        inputs = []
        for i in range(len(config.inputs_shape)):
            input_i = self.variable(
                name='input_' + str(i),
                shape=config.inputs_shape[i],
                dtype=config.inputs_dtype[i])
            inputs.append(input_i)
        result = paddle.add_n(inputs=inputs)

        self.feed_list = inputs
        self.fetch_list = [result]


class TorchAddN(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        input_0 = self.variable(
            name='input_' + str(0),
            shape=config.inputs_shape[0],
            dtype=config.inputs_dtype[0])
        result = input_0
        for i in range(1, len(config.inputs_shape)):
            input_i = self.variable(
                name='input_' + str(i),
                shape=config.inputs_shape[i],
                dtype=config.inputs_dtype[i])
            result = torch.add(result, input_i)
            self.feed_list = [input_i]

        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDAddN(), torch_obj=TorchAddN(), config=APIConfig("add_n"))
