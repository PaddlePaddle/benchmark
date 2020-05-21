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


class ReduceConfig(APIConfig):
    def __init__(self):
        super(ReduceConfig, self).__init__('reduce')
        self.api = 'mean'
        self.atol = 1e-5
        self.api_list = {'mean': 'mean', 'sum': 'sum', 'prod': 'prod'}

    def to_tensorflow(self):
        self.tf_api = self.api_list[self.api]
        return self


class PDReduce(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            data = fluid.data(
                name='input',
                shape=config.input_shape,
                dtype=config.input_dtype,
                lod_level=0)
            data.stop_gradient = False
            self.name = "reduce_" + config.api
            result = self.layers(
                "reduce_" + config.api,
                input=data,
                dim=config.dim,
                keep_dim=config.keep_dim)

            self.feed_vars = [data]
            self.fetch_vars = [result]
            if config.backward:
                self.append_gradients(result, [data])


class TFReduce(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        data = self.placeholder(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        self.name = "reduce_" + config.tf_api
        result = self.layers(
            "reduce_" + config.tf_api,
            input_tensor=data,
            axis=config.dim,
            keepdims=config.keep_dim)

        self.feed_list = [data]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [data])


if __name__ == '__main__':
    test_main(PDReduce(), TFReduce(), ReduceConfig())
