#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from resize_nearest import ResizeNearestConfig


class ResizeBilinearConfig(ResizeNearestConfig):
    def __init__(self):
        super(ResizeBilinearConfig, self).__init__("resize_bilinear")


class PDResizeBilinear(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        input = self.variable(
            name="input", shape=config.input_shape, dtype=config.input_dtype)
        result = fluid.layers.resize_bilinear(
            input=input,
            out_shape=config.out_shape,
            scale=config.scale,
            align_corners=True,
            align_mode=1,
            data_format=config.data_format)

        self.feed_vars = [input]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [input])


class TFResizeBilinear(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name="input", shape=config.input_shape, dtype=config.input_dtype)
        result = tf.image.resize(
            images=input,
            size=config.out_shape,
            method=tf.image.ResizeMethod.BILINEAR,
            preserve_aspect_ratio=False,
            antialias=False)

        self.feed_list = [input]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    test_main(
        PDResizeBilinear(), TFResizeBilinear(), config=ResizeBilinearConfig())
