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


class CastConfig(APIConfig):
    def __init__(self):
        super(CastConfig, self).__init__('cast')
        self.feed_spec = {"range": [-10, 10]}


class PDCast(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = paddle.cast(x=x, dtype=config.dtype)

        self.feed_list = [x]
        self.fetch_list = [result]


class TorchCast(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool
        }
        if config.dtype in dtype_map.keys():
            result = x.to(dtype_map[config.dtype])
        else:
            assert False, "Not supported yet!"

        self.feed_list = [x]
        self.fetch_list = [result]


if __name__ == '__main__':

    test_main(pd_dy_obj=PDCast(), torch_obj=TorchCast(), config=CastConfig())
