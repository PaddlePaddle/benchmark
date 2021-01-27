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


class UniqueConfig(APIConfig):
    def __init__(self):
        super(UniqueConfig, self).__init__("unique")
        # In version 1.7, unique API in torch has a bug to be fixed
        # "For some reasons, our torch.unique implementation is not differentiable."
        self.run_torch = False


class PDUnique(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name="x", shape=config.x_shape, dtype=config.x_dtype)
        result = paddle.unique(
            x=x,
            return_index=config.return_index,
            return_inverse=config.return_inverse,
            return_counts=config.return_counts,
            axis=config.axis,
            dtype=config.dtype)
        if isinstance(result, tuple):
            result = list(result)
        else:
            result = [result]
        self.feed_list = [x]
        self.fetch_list = result


# NOTE: pytorch has no return_index, 
# so the "return_index" is set "False" in config file
class TorchUnique(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name="x", shape=config.x_shape, dtype=config.x_dtype)
        result = torch.unique(
            input=x,
            sorted=True,
            return_inverse=config.return_inverse,
            return_counts=config.return_counts,
            dim=config.axis)

        self.feed_list = [x]
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDUnique(), torch_obj=TorchUnique(), config=UniqueConfig())
