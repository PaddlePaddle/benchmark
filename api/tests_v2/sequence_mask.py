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


class SequenceMaskConfig(APIConfig):
    def __init__(self):
        super(SequenceMaskConfig, self).__init__('sequence_mask')

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(SequenceMaskConfig, self).init_from_json(filename, config_id,
                                                       unknown_dim)
        if hasattr(self, "maxlen_shape"):
            self.maxlen = 3


class PDSequenceMask(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        data = self.variable(
            name='data', shape=config.x_shape, dtype=config.x_dtype)
        if hasattr(config, "maxlen_shape"):
            # maxlen is a Variable
            maxlen = paddle.full(
                shape=config.maxlen_shape,
                fill_value=config.maxlen,
                dtype=config.maxlen_dtype)
        else:
            maxlen = config.maxlen

        result = paddle.fluid.layers.sequence_mask(
            x=data, maxlen=maxlen, dtype=config.dtype)

        self.feed_vars = [data]
        self.fetch_vars = [result]


class TFSequenceMask(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        data = self.variable(
            name='data', shape=config.x_shape, dtype=config.x_dtype)

        # maxlen must be scalar for sequence_mask
        result = tf.sequence_mask(
            lengths=data,
            maxlen=config.maxlen,
            dtype=tf.as_dtype(config.dtype))

        self.feed_list = [data]
        self.fetch_list = [result]


if __name__ == '__main__':
    test_main(PDSequenceMask(), TFSequenceMask(), config=SequenceMaskConfig())