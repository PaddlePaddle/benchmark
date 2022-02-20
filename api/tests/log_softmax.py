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


@benchmark_registry.register("log_softmax")
class LogSoftmaxConfig(APIConfig):
    def __init__(self):
        super(LogSoftmaxConfig, self).__init__("log_softmax")
        self.feed_spec = {"range": [-1, 1]}
        # log_softmax is a combination of log and softmax, so that it can reuse softmax.json. 
        self.alias_name = "softmax"
        self.atol = 1E-5


@benchmark_registry.register("log_softmax")
class PaddleLogSoftmax(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        out = paddle.nn.functional.log_softmax(x=x, axis=config.axis)

        self.feed_list = [x]
        self.fetch_list = [out]
        if config.backward:
            self.append_gradients(out, [x])


@benchmark_registry.register("log_softmax")
class TorchLogSoftmax(PytorchOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        out = torch.nn.functional.log_softmax(input=x, dim=config.axis)

        self.feed_list = [x]
        self.fetch_list = [out]
        if config.backward:
            self.append_gradients(out, [x])


@benchmark_registry.register("log_softmax")
class TFLogSoftmax(TensorflowOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        out = tf.nn.log_softmax(logits=x, axis=config.axis)

        self.feed_list = [x]
        self.fetch_list = [out]
        if config.backward:
            self.append_gradients(out, [x])
