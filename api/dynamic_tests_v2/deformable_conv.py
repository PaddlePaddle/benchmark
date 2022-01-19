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
import torchvision


class DeformableConvConfig(APIConfig):
    def __init__(self, op_type="deformable_conv"):
        super(DeformableConvConfig, self).__init__(op_type)
        self.feed_spec = [
            {
                "range": [0, 1]
            },  # input
            {
                "range": [0, 1],
            },  # weight
            {
                "range": [0, 1],
            },  # offset
            {
                "range": [0, 1],
            }  # mask
        ]

    def to_pytorch(self):
        torch_config = super(DeformableConvConfig, self).to_pytorch()
        if isinstance(self.stride, int):
            torch_config.stride = [self.stride, self.stride]
        if isinstance(self.dilation, int):
            torch_config.dilation = [self.dilation, self.dilation]
        if isinstance(self.padding, int):
            torch_config.padding = [self.padding, self.padding]
        return torch_config


class PDDeformableConv(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name='weight',
            shape=config.weight_shape,
            dtype=config.weight_dtype)
        offset = self.variable(
            name='offset',
            shape=config.offset_shape,
            dtype=config.offset_dtype)
        mask = self.variable(
            name='mask', shape=config.mask_shape, dtype=config.mask_dtype)

        result = paddle.vision.ops.deform_conv2d(
            x=x,
            offset=offset,
            weight=weight,
            stride=config.stride,
            dilation=config.dilation,
            padding=config.padding,
            deformable_groups=config.deformable_groups,
            groups=config.groups,
            mask=mask)

        self.feed_list = [x, weight]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, weight])


class TorchDeformableConv(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name='x', shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name='weight', shape=config.weight_shape, dtype=config.x_dtype)
        offset = self.variable(
            name='offset',
            shape=config.offset_shape,
            dtype=config.offset_dtype)
        mask = self.variable(
            name='mask', shape=config.mask_shape, dtype=config.mask_dtype)
        result = torchvision.ops.deform_conv2d(
            input=input,
            offset=offset,
            weight=weight,
            stride=config.stride,
            padding=config.padding,
            dilation=config.dilation,
            mask=mask)

        self.feed_list = [input, weight]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input, weight])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDDeformableConv(),
        torch_obj=TorchDeformableConv(),
        config=DeformableConvConfig())
