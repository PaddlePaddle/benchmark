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


class ElementwiseConfig(APIConfig):
    def __init__(self):
        super(ElementwiseConfig, self).__init__('elementwise')
        self.api_list = ['add', 'sub', 'mul', 'pow', 'div']
        self.tf_api_list = ['add', 'subtract', 'multiply', 'pow', 'divide']
        self.id = 0

    def to_tensorflow(self):
        tf_config = self
        if self.act == "relu":
            tf_config.act = tf.nn.relu
        return tf_config


class PDElementwise(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            x = fluid.data(
                name='x',
                shape=config.x_shape,
                dtype=config.x_dtype,
                lod_level=0)
            y = fluid.data(
                name='y',
                shape=config.y_shape,
                dtype=config.y_dtype,
                lod_level=0)
            x.stop_gradient = False
            y.stop_gradient = False
            self.name = 'elementwise_' + config.api_list[config.id]
            module = importlib.import_module("paddle.fluid.layers")
            api_paddle = getattr(module,
                                 'elementwise_' + config.api_list[config.id])
            result = api_paddle(x=x, y=y, axis=config.axis, act=config.act)

            self.feed_vars = [x, y]
            self.fetch_vars = [result]
            if config.backward:
                self.append_gradients(result, [x, y])


class TFElementwise(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.placeholder(
            name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.placeholder(
            name='y', shape=config.y_shape, dtype=config.y_dtype)
        self.name = config.tf_api_list[config.id]
        module = importlib.import_module("tensorflow")
        api_tf = getattr(module, config.tf_api_list[config.id])
        result = api_tf(x=x, y=y)

        self.feed_list = [x, y]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, y])


if __name__ == '__main__':
    config = ElementwiseConfig()
    for i in range(len(config.api_list)):
        test_main(PDElementwise(), TFElementwise(), config)
        config.id = config.id + 1
