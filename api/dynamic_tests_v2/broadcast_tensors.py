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

class BroadcastTensorsConfig(APIConfig):
    def __init__(self):
        super(BroadcastTensorsConfig, self).__init__('broadcast_tensors')
        self.run_torch = False

class PDBroadcastTensors(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        inputs = []
        for i in range(len(config.input_shape)):
            input = self.variable(
                name='input_' + str(i),
                shape=config.input_shape[i],
                dtype=config.input_dtype[i])
            inputs.append(input)
        
        result = paddle.broadcast_tensors(input=inputs)

        self.feed_list = [inputs]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, inputs)

if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDBroadcastTensors(),
        config=BroadcastTensorsConfig())
