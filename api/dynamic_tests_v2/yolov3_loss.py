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


class Yolov3LossConfig(APIConfig):
    def __init__(self):
        super(Yolov3LossConfig, self).__init__("yolov3_loss")
        self.run_torch = False
        self.feed_spec = [{
            "range": [-1, 1]
        }, {
            "range": [0, 1]
        }, {
            "range": [0, 10]
        }, {
            "range": [0, 1]
        }]


class PDYolov3Loss(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        gt_box = self.variable(
            name='gt_box',
            shape=config.gt_box_shape,
            dtype=config.gt_box_dtype)
        gt_label = self.variable(
            name='gt_label',
            shape=config.gt_label_shape,
            dtype=config.gt_label_dtype)
        gt_score = self.variable(
            name='gt_score',
            shape=config.gt_score_shape,
            dtype=config.gt_score_dtype)

        result = paddle.fluid.layers.yolov3_loss(
            x=x,
            gt_box=gt_box,
            gt_label=gt_label,
            gt_score=gt_score,
            anchors=config.anchors,
            anchor_mask=config.anchor_mask,
            class_num=config.class_num,
            ignore_thresh=config.ignore_thresh,
            downsample_ratio=config.downsample_ratio)

        self.feed_list = [x, gt_box, gt_label, gt_score]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(pd_dy_obj=PDYolov3Loss(), config=Yolov3LossConfig())
