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


class PDInterpLinear(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        out = paddle.nn.functional.interpolate(
            x,
            size=config.size,
            mode="linear",
            align_corners=config.align_corners,
            scale_factor=config.scale_factor,
            data_format=config.data_format)
        self.feed_list = [x]
        self.fetch_list = [out]
        if config.backward:
            self.append_gradients(out, [x])

    def compute_flop_and_byte(self, config):
        # at least one of out_shape and scale must be set 
        x_shape = config.x_shape
        out_size = config.size

        assert (config.scale_factor is not None or out_size is not None
                ), "at least one of out_shape and scale must be set"
        # config.size has higher priority than config.scale_factor
        if isinstance(out_size, (list, tuple)):
            out_shape = x_shape[0:-len(out_size)] + out_size
        elif isinstance(config.scale_factor, (list, tuple)):
            scale_length = len(config.scale_factor)
            change_out = x_shape[-scale_length:]
            scale_out = [
                i * j for i, j in zip(change_out, config.scale_factor)
            ]
            out_shape = x_shape[0:-scale_length] + scale_out
        elif isinstance(config.scale_factor, float):
            change_out = x_shape[-1:]
            scale_out = [i * config.scale_factor for i in change_out]
            out_shape = x_shape[0:-1] + scale_out

        # forward flops, sub*4 + mul*2 + div*2 + add*1
        forward_flop = numel(out_shape) * 9

        # forward byte, read 5 address to compute 1 address
        read_byte = 5 * numel(out_shape) * sizeof(config.x_dtype)
        write_byte = numel(out_shape) * sizeof(config.x_dtype)
        forward_byte = read_byte + write_byte
        if not config.backward:
            return forward_flop, forward_byte
        else:
            # to be implemented.
            return None, None


class TorchInterpLinear(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = torch.nn.functional.interpolate(
            x,
            size=config.size,
            mode="linear",
            align_corners=config.align_corners,
            scale_factor=config.scale_factor)
        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDInterpLinear(),
        torch_obj=TorchInterpLinear(),
        config=APIConfig("interp_linear"))
