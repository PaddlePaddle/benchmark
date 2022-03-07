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


@benchmark_registry.register("activation")
class ActivationConfig(APIConfig):
    def __init__(self):
        super(ActivationConfig, self).__init__('activation')
        self.api_name = 'sigmoid'
        self.api_list = {
            'sigmoid': 'sigmoid',
            'relu': 'relu',
            'relu6': 'relu6',
            'leaky_relu': 'leaky_relu',
            'elu': 'elu',
            'hardsigmoid': 'hardsigmoid',
            'hardswish': 'hardswish',
            'selu': 'selu',
            'softplus': 'softplus',
            'tanhshrink': 'tanhshrink',
            'softshrink': 'softshrink',
            'softsign': 'softsign'
        }

    @property
    def run_tf(self):
        if self.api_name in [
                "hardsigmoid", "hardswish", "tanhshrink", "softshrink"
        ]:
            print("-- %s is not supported in tf." % self.api_name)
            return False
        return True

    def disabled(self):
        if self.api_name in ["selu"] and self.x_dtype == "float16":
            print(
                "-- Warning:\n"
                "  1. This config is disabled because float16 is not supported for %s.\n"
                % (self.api_name))
            return True
        return super(ActivationConfig, self).disabled()


@benchmark_registry.register("activation")
class PaddleActivation(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = self.layers(
            config.api_name, module_name="paddle.nn.functional", x=x)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


@benchmark_registry.register("activation")
class TorchActivation(PytorchOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = self.layers(
            config.api_name, module_name="torch.nn.functional", input=x)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


@benchmark_registry.register("activation")
class TFActivation(TensorflowOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = self.layers(config.api_name, x=x)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])
