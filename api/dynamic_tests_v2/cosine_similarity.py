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


class PDCosineSimilarity(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x1 = self.variable(
            name='x1', shape=config.x1_shape, dtype=config.x1_dtype)
        x2 = self.variable(
            name='x2', shape=config.x2_shape, dtype=config.x2_dtype)
        result = paddle.nn.functional.cosine_similarity(
            x1=x1, x2=x2, axis=config.axis, eps=1e-8)

        self.feed_list = [x1, x2]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x1, x2])


class TorchCosineSimilarity(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x1 = self.variable(
            name='x1', shape=config.x1_shape, dtype=config.x1_dtype)
        x2 = self.variable(
            name='x2', shape=config.x2_shape, dtype=config.x2_dtype)

        result = torch.cosine_similarity(
            x1=x1, x2=x2, dim=config.axis, eps=1e-8)

        self.feed_list = [x1, x2]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x1, x2])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDCosineSimilarity(),
        torch_obj=TorchCosineSimilarity(),
        config=APIConfig("cosine_similarity"))
