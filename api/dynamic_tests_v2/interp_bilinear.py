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


class InterpBilinearConfig(APIConfig):
    def __init__(self):
        super(InterpBilinearConfig, self).__init__('interp_bilinear')

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(InterpBilinearConfig, self).init_from_json(filename, config_id,
                                                         unknown_dim)
        if self.data_format == 'NHWC':
            print(
                "Warning:\n"
                "  1. PyTorch does not have data_format param, it only support NCHW format.\n"
            )
            self.run_torch = False


class PDInterpBilinear(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        out = paddle.nn.functional.interpolate(
            x,
            size=config.size,
            mode="bilinear",
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
        # scale_factor shouldn`t to be a float in bilinear mode
        elif isinstance(config.scale_factor, (list, tuple)):
            scale_length = len(config.scale_factor)
            change_out = x_shape[-scale_length:]
            scale_out = [
                i * j for i, j in zip(change_out, config.scale_factor)
            ]
            out_shape = x_shape[0:-scale_length] + scale_out

        # forward flops, sub*10 + mul*9 + div*1 + add*3
        forward_flop = numel(out_shape) * 23

        # forward byte, read 4 address to compute 1 address
        read_byte = 4 * numel(out_shape) * sizeof(config.x_dtype)
        write_byte = numel(out_shape) * sizeof(config.x_dtype)
        forward_byte = read_byte + write_byte
        if not config.backward:
            return forward_flop, forward_byte
        else:
            # to be implemented.
            return None, None


class TorchInterpBilinear(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = torch.nn.functional.interpolate(
            x,
            size=config.size,
            mode="bilinear",
            align_corners=config.align_corners,
            scale_factor=config.scale_factor)
        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDInterpBilinear(),
        torch_obj=TorchInterpBilinear(),
        config=InterpBilinearConfig())
