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

    def disabled(self):
        if not use_gpu():
            print(
                "Warning:\n"
                "  1. lstm is disabled because it is only supported run on GPU, now is CPU.\n"
            )
            return True
        return super(LstmConfig, self).disabled()


class PDLstm(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        input = self.variable(
            name="input", shape=config.inputs_shape, dtype=config.inputs_dtype)

        init_h = paddle.full(
            shape=config.inital_states_shape,
            dtype=config.inital_states_dtype,
            fill_value=0.0)
        init_c = paddle.full(
            shape=config.inital_states_shape,
            dtype=config.inital_states_dtype,
            fill_value=0.0)

        rnn = paddle.nn.LSTM(
            input_size=config.inputs_shape[-1],
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=0.0,
            direction=config.direction)

        rnn_out, (last_h, last_c) = rnn(input, (init_h, init_c))

        self.feed_vars = [input]
        self.fetch_vars = [rnn_out]


class TFLstm(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name="input", shape=config.inputs_shape, dtype=config.inputs_dtype)

        init_h = tf.constant(
            shape=config.inital_states_shape,
            dtype=tf.as_dtype(config.inital_states_dtype),
            value=0.0)
        init_c = tf.constant(
            shape=config.inital_states_shape,
            dtype=tf.as_dtype(config.inital_states_dtype),
            value=0.0)


if __name__ == '__main__':
    test_main(PDLstm(), TFLstm(), config=LstmConfig())
