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


class PaddleEye(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        result = paddle.eye(num_rows=config.num_rows, num_columns=config.num_columns, dtype=config.dtype)

        self.feed_list = []
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [])


class TorchEye(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool
        }
        result = torch.eye(n=config.num_rows, m=config.num_columns, dtype=dtype_map[config.dtype])

        self.feed_list = []
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PaddleEye(),
        torch_obj=TorchEye(),
        config=APIConfig('eye'))
