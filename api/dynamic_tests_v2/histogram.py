#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


class HistogramConfig(APIConfig):
    def __init__(self):
        super(HistogramConfig, self).__init__("histogram")
        self.feed_spec = [{"range": [-20, 20]}]

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(HistogramConfig, self).init_from_json(filename, config_id,
                                                    unknown_dim)

        if not use_gpu() and self.input_dtype in ["int32", "int64"]:
            print(
                "Warning:\n"
                "1. Pytorch-CPU can not support the histogram curently once the input data dtype is"
                " int32 or int64\n")
            self.run_torch = False


class PDHistogram(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        result = paddle.histogram(
            input=x, bins=config.bins, min=config.min, max=config.max)

        self.feed_list = [x]
        self.fetch_list = [result]


class TorchHistogram(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        result = torch.histc(
            input=x, bins=config.bins, min=config.min, max=config.max)

        self.feed_list = [x]
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDHistogram(),
        torch_obj=TorchHistogram(),
        config=HistogramConfig())
