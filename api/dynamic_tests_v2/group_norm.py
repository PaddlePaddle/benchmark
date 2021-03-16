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


class GroupNormConfig(APIConfig):
    def __init__(self):
        super(GroupNormConfig, self).__init__('group_norm')

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(GroupNormConfig, self).init_from_json(filename, 
                                        config_id, unknown_dim)
    # num_channels
    if len(self.x_shape) == 4:
        self.num_channels = self.x_shape[1] \
            if self.data_format == "NCHW" else self.x_shape[3]
    else :
        self.num_channels= self.x_shape[1] 

    self._set_param_dtype()
    if self.data_layout != 'NCHW' :
        raise ValueError(
        "Param(data_layout) of Op(fluid.layers.group_norm) got wrong value: "
        "received " + data_layout + " but only NCHW or NHWC supported.")

    def _set_param_dtype(self):
        # dtype of parameters
        self.param_dtype = "float32" if self.x_dtype == "float16" else self.x_dtype


class PDGroupNorm(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = paddle.fluid.layers.group_norm(
            x=x,
            epsilon=config.epsilon,
            num_channels=config.num_channels,
            num_groups=config.num_groups)

        self.feed_list = [x, result]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TorchGroupNorm(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = torch.nn.functional.group_norm(
            input=x,
            num_groups=config.num_groups,
            num_channels=config.num_channels,
            eps=config.epsilon)

        self.feed_list = [x, result]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDGroupNorm(),
        torch_obj=TorchGroupNorm(),
        config=GroupNormConfig("group_norm"))
