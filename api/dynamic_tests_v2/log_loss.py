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


class LogLossConfig(APIConfig):
    def __init__(self):
        super(LogLossConfig, self).__init__("log_loss")
        self.run_torch = False
        self.feed_spec = [{"range": [-1, 1]}, {"range": [-1, 1]}]


class PDLogLoss(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name='input', shape=config.input_shape, dtype=config.input_dtype)
        label = self.variable(
            name='label', shape=config.label_shape, dtype=config.label_dtype)
        result = paddle.nn.functional.log_loss(
            input=input, label=label, epsilon=config.epsilon)

        self.feed_list = [input, label]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [input])


if __name__ == '__main__':
    test_main(pd_dy_obj=PDLogLoss(), config=LogLossConfig())
