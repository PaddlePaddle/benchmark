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


class LogSoftmaxConfig(APIConfig):
    def __init__(self):
        super(LogSoftmaxConfig, self).__init__("log_softmax")
        self.feed_spec = {"range": [-1, 1]}
        # log_softmax is a combination of log and softmax, so that it can reuse softmax.json. 
        self.alias_name = "softmax"


class PDLogSoftmax(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        out = paddle.nn.functional.log_softmax(x=x, axis=config.axis)

        self.feed_list = [x]
        self.fetch_list = [out]
        if config.backward:
            self.append_gradients(out, [x])

    def compute_flop_and_byte(self, config):
        x_shape = config.x_shape
        # log((exp(x[i, j]) - max(x[i, j])) / sum_j(exp(x[i, j]) - max(x[i, j]))) =
        # (x[i, j] - max(x[i, j])) - log(sum_j(exp(x[i, j] - max(i, j))))
        softmax_dim = x_shape[config.axis]
        forward_flop = numel(x_shape) * 5 + numel(x_shape) / softmax_dim
        forward_byte = numel(x_shape) * 2 * sizeof(config.x_dtype)
        if not config.backward:
            return forward_flop, forward_byte
        else:
            # To be implemented.
            return None, None


class TorchLogSoftmax(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        out = torch.nn.functional.log_softmax(input=x, dim=config.axis)

        self.feed_list = [x]
        self.fetch_list = [out]
        if config.backward:
            self.append_gradients(out, [x])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDLogSoftmax(),
        torch_obj=TorchLogSoftmax(),
        config=LogSoftmaxConfig())
