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


class YoloBoxConfig(APIConfig):
    def __init__(self):
        super(YoloBoxConfig, self).__init__("yolo_box")
        self.run_tf = False


class PDYoloBox(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        img_size = self.variable(
            name='img_size',
            shape=config.img_size_shape,
            dtype=config.img_size_dtype)
        boxes, scores = paddle.fluid.layers.yolo_box(
            x=x,
            img_size=img_size,
            class_num=config.class_num,
            anchors=config.anchors,
            conf_thresh=config.conf_thresh,
            downsample_ratio=config.downsample_ratio)

        self.feed_vars = [x, img_size]
        self.fetch_vars = [boxes, scores]


if __name__ == '__main__':
    test_main(PDYoloBox(), config=YoloBoxConfig())
