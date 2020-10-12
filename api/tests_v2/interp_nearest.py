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


class InterpNearestConfig(APIConfig):
    def __init__(self):
        super(InterpNearestConfig, self).__init__("interp_nearest")
        self.run_tf = False


class PDInterpNearest(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        # align_corners option can only be set with the interpolating modes: linear, bilinear, bicubic, trilinear
        out = paddle.nn.functional.interpolate(
            x=x,
            size=config.size,
            mode="nearest",
            align_corners=config.align_corners,
            scale_factor=config.scale_factor,
            data_format=config.data_format)
        self.feed_vars = [x]
        self.fetch_vars = [out]
        if config.backward:
            self.append_gradients(out, [x])


if __name__ == '__main__':
    test_main(PDInterpNearest(), config=InterpNearestConfig())
