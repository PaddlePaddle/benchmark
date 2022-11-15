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


@benchmark_registry.register("interp_bilinear")
class InterpBilinearConfig(APIConfig):
    def __init__(self):
        super(InterpBilinearConfig, self).__init__('interp_bilinear')
        self.run_tf = False

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(InterpBilinearConfig, self).init_from_json(filename, config_id,
                                                         unknown_dim)
        if self.data_format == 'NHWC':
            print(
                "Warning:\n"
                "  1. PyTorch does not have data_format param, it only support NCHW format.\n"
            )
            self.run_torch = False


@benchmark_registry.register("interp_bilinear")
class PaddleInterpBilinear(PaddleOpBenchmarkBase):
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
        x_shape = config.x_shape
        out_shape = self.fetch_list[0].shape
        # forward flop, sub * 10 + mul * 9 + div * 1 + add * 3
        forward_flop = numel(out_shape) * 23
        forward_byte = (
            numel(x_shape) + numel(out_shape)) * sizeof(config.x_dtype)
        if not config.backward:
            return forward_flop, forward_byte
        else:
            # to be implemented.
            return None, None


@benchmark_registry.register("interp_bilinear")
class TorchInterpBilinear(PytorchOpBenchmarkBase):
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
