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


class LstmConfig(APIConfig):
    def __init__(self):
        super(LstmConfig, self).__init__("lstm")
        self.run_tf = False


class PDLstm(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        input = self.variable(
            name="input", shape=config.input_shape, dtype=config.input_dtype)

        init_h = fluid.layers.fill_constant(
            shape=config.init_h_shape, dtype=config.init_h_dtype, value=0.0)
        init_c = fluid.layers.fill_constant(
            shape=config.init_c_shape, dtype=config.init_c_dtype, value=0.0)

        rnn_out, last_h, last_c = fluid.layers.lstm(
            input=input,
            init_h=init_h,
            init_c=init_c,
            max_len=config.max_len,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout_prob=0.2,
            is_bidirec=config.is_bidirec)

        self.feed_vars = [input]
        self.fetch_vars = [rnn_out]
        if config.backward:
            self.append_gradients(rnn_out, [input])


class TFLstm(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name="input", shape=config.input_shape, dtype=config.input_dtype)

        init_h = tf.constant(
            shape=config.init_h_shape,
            dtype=tf.as_dtype(config.init_h_dtype),
            value=0.0)
        init_c = tf.constant(
            shape=config.init_c_shape,
            dtype=tf.as_dtype(config.init_c_dtype),
            value=0.0)


if __name__ == '__main__':
    test_main(PDLstm(), TFLstm(), config=LstmConfig())
