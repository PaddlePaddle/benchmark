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

import copy

from common_import import *


class ElementwiseConfig(APIConfig):
    def __init__(self):
        super(ElementwiseConfig, self).__init__('elementwise')
        self.api_name = 'elementwise_add'
        self.api_list = {
            'elementwise_add': 'add',
            'elementwise_div': 'divide',
            'elementwise_max': 'maximum',
            'elementwise_min': 'minimum',
            'elementwise_sub': 'subtract',
            'elementwise_mul': 'multiply',
            'elementwise_pow': 'pow'
        }
        self.feed_spec = [{"range": [-1, 1]}, {"range": [-1, 1]}]

    def disabled(self):
        if self.api_name in [
                "elementwise_max", "elementwise_min", "elementwise_pow"
        ] and self.x_dtype == "float16":
            return True
        return False

    def to_tensorflow(self):
        tf_config = super(ElementwiseConfig, self).to_tensorflow()
        if len(self.x_shape) > len(self.y_shape) and self.y_shape != [1]:
            tf_config.y_shape_unsqueezed = self._unsqueeze_short(
                short=self.y_shape, long=self.x_shape)
        elif len(self.x_shape) < len(self.y_shape) and self.x_shape != [1]:
            tf_config.x_shape_unsqueezed = self._unsqueeze_short(
                short=self.x_shape, long=self.y_shape)
        return tf_config

    def _unsqueeze_short(self, short, long):
        short_extend = np.ones([len(long)], dtype=np.int32).tolist()
        start = 0
        for value in short:
            for i in range(start, len(long)):
                if long[i] == value:
                    short_extend[i] = value
                    start = i
                    break
        return short_extend


class PDElementwise(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)
        result = self.fluid_layers(
            config.api_name, x=x, y=y, axis=config.axis, act=None)

        self.feed_vars = [x, y]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x, y])


class TFElementwise(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.variable(name='y', shape=config.y_shape, dtype=config.y_dtype)
        if hasattr(config, "x_shape_unsqueezed"):
            x_reshape = tf.reshape(tensor=x, shape=config.x_shape_unsqueezed)
        else:
            x_reshape = x
        if hasattr(config, "y_shape_unsqueezed"):
            y_reshape = tf.reshape(tensor=y, shape=config.y_shape_unsqueezed)
        else:
            y_reshape = y
        result = self.layers(config.api_name, x=x_reshape, y=y_reshape)

        self.feed_list = [x, y]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, y])


if __name__ == '__main__':
    test_main(PDElementwise(), TFElementwise(), config=ElementwiseConfig())
