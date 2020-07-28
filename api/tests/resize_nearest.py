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


class ResizeNearestConfig(APIConfig):
    def __init__(self, op_type="resize_nearest"):
        super(ResizeNearestConfig, self).__init__(op_type)

    def init_from_json(self, filename, config_id=0):
        super(ResizeNearestConfig, self).init_from_json(filename, config_id)
        if self.data_format == "NCHW":
            # tf only support NHWC
            if len(self.input_shape) == 4:
                self.feed_spec = {"permute": [0, 2, 3, 1]}
            elif len(self.input_shape) == 3:
                self.feed_spec = {"permute": [1, 2, 0]}

    def to_tensorflow(self):
        tf_config = super(ResizeNearestConfig, self).to_tensorflow()
        if self.data_format == "NCHW":
            # tf only support NHWC
            if len(self.input_shape) == 4:
                tf_config.input_shape = [
                    self.input_shape[0], self.input_shape[2],
                    self.input_shape[3], self.input_shape[1]
                ]
            elif len(self.input_shape) == 3:
                tf_config.input_shape = [
                    self.input_shape[1], self.input_shape[2],
                    self.input_shape[0]
                ]
        return tf_config


class PDResizeNearest(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        input = self.variable(
            name="input", shape=config.input_shape, dtype=config.input_dtype)
        result = fluid.layers.resize_nearest(
            input=input,
            out_shape=None,
            scale=config.scale,
            align_corners=True,
            data_format=config.data_format)

        self.feed_vars = [input]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [input])


class TFResizeNearest(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name="input", shape=config.input_shape, dtype=config.input_dtype)
        result = tf.image.resize(
            images=input,
            size=config.out_shape,
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            preserve_aspect_ratio=False)

        self.feed_list = [input]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    test_main(
        PDResizeNearest(), TFResizeNearest(), config=ResizeNearestConfig())
