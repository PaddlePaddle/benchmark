##   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

    def init_from_json(self, filename, config_id=0, unkown_dim=16):
        super(LstmConfig, self).init_from_json(filename, config_id, unkown_dim)
        if self.direction == "bidirect":
            self.inital_states_shape[0] *= 2
            self.inital_states_shape[0] *= 2

    def disabled(self):
        if not use_gpu():
            print(
                "Warning:\n"
                "  1. lstm is disabled because it is only supported run on GPU, now is CPU.\n"
            )
            return True
        return super(LstmConfig, self).disabled()

    def to_pytorch(self):
        torch_config = super(LstmConfig, self).to_pytorch()
        torch_config.inputs_shape = [
            self.inputs_shape[1], self.inputs_shape[0], self.inputs_shape[2]
        ]

        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool
        }
        torch_config.inital_states_dtype = dtype_map[self.inital_states_dtype]

        torch_config.direction_type = "bool"
        if self.direction == "forward":
            torch_config.direction = False
        else:
            torch_config.direction = True
        return torch_config


class PDLstm(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
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

        self.feed_list = [input]
        self.fetch_list = [rnn_out]


class TorchLstm(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name="input", shape=config.inputs_shape, dtype=config.inputs_dtype)

        # tensor_h = torch.empty(
        #     config.inital_states_shape, dtype=config.inital_states_dtype)
        # init_h = torch.nn.init.constant_(tensor=tensor_h, val=0.0)
        init_h = torch.full(
            size=config.inital_states_shape,
            fill_value=0.0,
            dtype=config.inital_states_dtype)

        # tensor_c = torch.empty(
        #     config.inital_states_shape, dtype=config.inital_states_dtype)
        # init_c = torch.nn.init.constant_(tensor=tensor_c, val=0.0)
        init_c = torch.full(
            size=config.inital_states_shape,
            fill_value=0.0,
            dtype=config.inital_states_dtype)

        rnn = torch.nn.LSTM(
            input_size=config.inputs_shape[-1],
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=0.0,
            batch_first=False,
            bidirectional=config.direction)
        # move model and input to cuda device
        if torch.cuda.is_available():
            rnn.cuda()
            input = input.cuda()
            init_h = init_h.cuda()
            init_c = init_c.cuda()

        rnn_out, (last_h, last_c) = rnn(input, (init_h, init_c))

        self.feed_list = [input]
        self.fetch_list = [rnn_out]


if __name__ == '__main__':
    test_main(pd_dy_obj=PDLstm(), torch_obj=TorchLstm(), config=LstmConfig())
