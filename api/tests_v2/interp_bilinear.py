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
        super(InterpBilinearConfig, self).__init__("interp_bilinear")
        self.run_tf = False


class PDInterpBilinear(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        out = paddle.nn.functional.interpolate(
            x,
            size=config.size,
            mode="bilinear",
            align_corners=config.align_corners,
            scale_factor=config.scale_factor,
            data_format=config.data_format)
        self.feed_vars = [x]
        self.fetch_vars = [out]
        if config.backward:
            self.append_gradients(out, [x])


if __name__ == '__main__':
    test_main(PDInterpBilinear(), config=InterpBilinearConfig())
