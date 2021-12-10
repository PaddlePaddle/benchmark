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
from reduce import PDReduce, TorchReduce


class ReduceAllAnyConfig(APIConfig):
    def __init__(self):
        super(ReduceAllAnyConfig, self).__init__('reduce_all_any')
        self.feed_spec = {"range": [-1, 1]}
        self.api_name = 'all'
        self.api_list = {'all': 'all', 'any': 'any'}

    def init_from_json(self, filename, config_id=3, unknown_dim=16):
        super(ReduceAllAnyConfig, self).init_from_json(filename, config_id,
                                                       unknown_dim)
        if self.axis == None:
            self.axis = []

    def squeeze_shape(self):
        if len(self.axis) == 1:
            return self.axis[0], self.x_shape
        elif len(self.axis) > 1:
            is_continuous = True
            for i in range(1, len(self.axis)):
                if self.axis[i] != self.axis[i - 1] + 1:
                    is_continuous = False
                    break
            if is_continuous:
                begin_axis = self.axis[0]
                end_axis = self.axis[-1]
                new_x_shape = [1] * (len(self.x_shape) + begin_axis - end_axis)
                for i in range(len(self.x_shape)):
                    if i < begin_axis:
                        new_x_shape[i] = self.x_shape[i]
                    elif i < end_axis + 1:
                        new_x_shape[begin_axis] = new_x_shape[
                            begin_axis] * self.x_shape[i]
                    else:
                        new_x_shape[i + begin_axis - end_axis] = self.x_shape[
                            i]
                return begin_axis, new_x_shape
        return self.axis, self.x_shape


class TorchReduceAllAny(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        axis, x_shape = config.squeeze_shape()
        if x_shape != config.x_shape:
            x = torch.reshape(x=x, shape=x_shape)
        if isinstance(axis, list) and len(axis) == 0:
            result = self.layers(config.api_name, input=x)
        else:
            result = self.layers(
                config.api_name, input=x, dim=axis, keepdim=config.keepdim)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDReduce(),
        torch_obj=TorchReduceAllAny(),
        config=ReduceAllAnyConfig())
