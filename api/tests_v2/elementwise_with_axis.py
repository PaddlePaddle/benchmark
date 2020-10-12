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


class ElementwiseWithAxisConfig(APIConfig):
    def __init__(self):
        super(ElementwiseWithAxisConfig,
              self).__init__('elementwise_with_axis')
        self.alias_config = APIConfig("elementwise")
        self.api_name = 'maximum'
        self.api_list = {
            'maximum': 'maximum',
            'minimum': 'minimum',
            'multiply': 'multiply',
            'elementwise_sub': 'subtract'
        }
        self.feed_spec = [{"range": [-1, 1]}, {"range": [-1, 1]}]

    def disabled(self):
        if self.api_name in ["maximum", "minimum"
                             ] and self.alias.x_dtype == "float16":
            return True
        return False

    def to_tensorflow(self):
        tf_config = super(ElementwiseWithAxisConfig, self).to_tensorflow()
        if self.api_name == 'multiply':
            self.atol = 1e-5
        else:
            self.atol = 1e-6
        if len(self.alias.x_shape) > len(
                self.alias.y_shape) and self.alias.y_shape != [1]:
            tf_config.y_shape_unsqueezed = self._unsqueeze_short(
                short=self.alias.y_shape, long=self.alias.x_shape)
        elif len(self.alias.x_shape) < len(
                self.alias.y_shape) and self.alias.x_shape != [1]:
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


class PDElementwiseWithAxis(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(
            name='x', shape=config.alias.x_shape, dtype=config.alias.x_dtype)
        y = self.variable(
            name='y', shape=config.alias.y_shape, dtype=config.alias.y_dtype)
        result = self.layers(config.api_name, x=x, y=y, axis=config.alias.axis)

        self.feed_vars = [x, y]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x, y])


class TFElementwiseWithAxis(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(
            name='x', shape=config.alias.x_shape, dtype=config.alias.x_dtype)
        y = self.variable(
            name='y', shape=config.alias.y_shape, dtype=config.alias.y_dtype)
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
    test_main(
        PDElementwiseWithAxis(),
        TFElementwiseWithAxis(),
        config=ElementwiseWithAxisConfig())
