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


class MaskedSelectConfig(APIConfig):
    def __init__(self):
        super(MaskedSelectConfig, self).__init__("masked_select")

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(MaskedSelectConfig, self).init_from_json(filename, config_id,
                                                       unknown_dim)
        # according to 'class PytorchAPIBenchmarkBase', in fucntion 'Variable'
        # requires_grad=False, when x_dtype==int. However, this API requires 
        # GRAD, so we do the following cast:
        if self.x_dtype == "int32":
            self.x_dtype = "float32"


class PDMaskedSelect(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name="x", shape=config.x_shape, dtype=config.x_dtype)
        mask = self.variable(
            name="mask", shape=config.mask_shape, dtype=config.mask_dtype)
        result = paddle.masked_select(x, mask)

        self.feed_list = [x, mask]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TorchMaskedSelect(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name="x", shape=config.x_shape, dtype=config.x_dtype)
        mask = self.variable(
            name="mask", shape=config.mask_shape, dtype=config.mask_dtype)
        result = torch.masked_select(input=x, mask=mask)

        self.feed_list = [x, mask]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDMaskedSelect(),
        torch_obj=TorchMaskedSelect(),
        config=MaskedSelectConfig())
