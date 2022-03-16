#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class MultiplexConfig(APIConfig):
    def __init__(self):
        super(MultiplexConfig, self).__init__('multiplex')
        self.run_torch = False
        print("[WARNING]: Pytorch dosen`t support multiplex currently.")

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(MultiplexConfig, self).init_from_json(filename, config_id,
                                                    unknown_dim)

        index = np.array([[3], [0], [1], [2]])
        self.index_value = index.astype(self.index_dtype)


class PaddleMultiplex(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        # The input of multiplex op is a list
        inputs = []
        for i in range(len(config.inputs_shape)):
            input = self.variable(
                name='input' + str(i),
                shape=config.inputs_shape[i],
                dtype=config.inputs_dtype[i])
            inputs.append(input)
        index = self.variable(
            name='index',
            shape=config.index_shape,
            dtype=config.index_dtype,
            value=config.index_value)
        result = paddle.multiplex(inputs=inputs, index=index)

        self.feed_list = [inputs, index]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, inputs)


if __name__ == '__main__':
    test_main(pd_dy_obj=PaddleMultiplex(), config=MultiplexConfig())
