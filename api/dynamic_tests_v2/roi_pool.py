# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


class ROIPoolConfig(APIConfig):
    def __init__(self):
        super(ROIPoolConfig, self).__init__("roi_pool")
        self.run_torch = False

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(ROIPoolConfig, self).init_from_json(filename, config_id,
                                                  unknown_dim)
        boxes = np.random.rand(960, 4)
        boxes[:, 2] += boxes[:, 0] + 50
        boxes[:, 3] += boxes[:, 1] + 50
        self.boxes_value = boxes
        self.boxes_num_value = np.array([960 // unknown_dim] *
                                        unknown_dim).astype('int32')


class PDROIPool(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        boxes = self.variable(
            name='boxes',
            shape=config.boxes_shape,
            dtype=config.boxes_dtype,
            value=config.boxes_value)
        boxes_num = self.variable(
            name='boxes_num',
            shape=config.boxes_num_shape,
            dtype=config.boxes_num_dtype,
            value=config.boxes_num_value)

        result = paddle.vision.ops.roi_pool(x, boxes, boxes_num,
                                            config.output_size)

        self.feed_vars = [x, boxes, boxes_num]
        self.fetch_vars = [result]


if __name__ == "__main__":
    test_main(pd_dy_obj=PDROIPool(), config=ROIPoolConfig())
