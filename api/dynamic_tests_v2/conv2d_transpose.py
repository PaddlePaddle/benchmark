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


class PDConv2dTranspose(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name="x", shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name="weight",
            shape=config.weight_shape,
            dtype=config.weight_dtype)

        result = paddle.nn.functional.conv2d_transpose(
            x=x,
            weight=weight,
            bias=None,
            stride=config.stride,
            padding=config.padding,
            output_padding=0,
            dilation=config.dilation,
            groups=1,
            output_size=config.output_size,
            data_format=config.data_format)

        self.feed_list = [x, weight]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, weight])


class TorchConv2dTranspose(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name="x", shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name="weight",
            shape=config.weight_shape,
            dtype=config.weight_dtype)

        result = torch.nn.functional.conv_transpose2d(
            input=x,
            weight=weight,
            bias=None,
            stride=config.stride,
            padding=config.padding,
            output_padding=0,
            dilation=config.dilation,
            groups=1)

        self.feed_list = [x, weight]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, weight])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDConv2dTranspose(),
        torch_obj=TorchConv2dTranspose(),
        config=APIConfig("conv2d_transpose"))
