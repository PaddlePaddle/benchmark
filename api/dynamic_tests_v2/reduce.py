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
        self.feed_spec = {"range": [-1, 1]}
        self.api_name = 'sum'
        self.api_list = {'sum': 'sum', 'mean': 'mean'}

    def disabled(self):
        if self.api_name == "mean" and self.x_dtype == "float16":
            print(
                "Warning:\n"
                "  1. This config is disabled because float16 is not supported for %s.\n"
                % (self.api_name))
            return True
        return super(ReduceConfig, self).disabled()

    def init_from_json(self, filename, config_id=3, unknown_dim=16):
        super(ReduceConfig, self).init_from_json(filename, config_id,
                                                 unknown_dim)
        if self.axis == None:
            self.axis = []


class PDReduce(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = self.layers(
            config.api_name, x=x, axis=config.axis, keepdim=config.keepdim)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])

    def compute_flop_and_byte(self, config):
        x_shape = config.x_shape
        out_shape = self.fetch_list[0].shape
        forward_flop = numel(x_shape)
        forward_byte = (
            numel(x_shape) + numel(out_shape)) * sizeof(config.x_dtype)
        if not config.backward:
            return forward_flop, forward_byte
        else:
            # To be implemented.
            return None, None


class TorchReduce(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        if isinstance(config.axis, list) and len(config.axis) == 0:
            result = self.layers(config.api_name, input=x)
        else:
            result = self.layers(
                config.api_name,
                input=x,
                dim=config.axis,
                keepdim=config.keepdim)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDReduce(), torch_obj=TorchReduce(), config=ReduceConfig())
