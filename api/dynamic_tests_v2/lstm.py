#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
        if self.is_bidirec:
            self.init_h_shape[0] *= 2
            self.init_c_shape[0] *= 2

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
        torch_config.input_shape = [
            self.input_shape[1], self.input_shape[0], self.input_shape[2]
        ]
        return torch_config


class PDLstm(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name="input", shape=config.input_shape, dtype=config.input_dtype)

        init_h = paddle.full(
            shape=config.init_h_shape,
            dtype=config.init_h_dtype,
            fill_value=0.0)
        init_c = paddle.full(
            shape=config.init_c_shape,
            dtype=config.init_c_dtype,
            fill_value=0.0)

        rnn_out, last_h, last_c = paddle.fluid.layers.lstm(
            input=input,
            init_h=init_h,
            init_c=init_c,
            max_len=config.max_len,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout_prob=0.2,
            is_bidirec=config.is_bidirec)

        self.feed_list = [input]
        self.fetch_list = [rnn_out]
        #if config.backward:
        #    self.append_gradients(rnn_out, [input])


class TorchLstm(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool
        }
        input = self.variable(
            name="input", shape=config.input_shape, dtype=config.input_dtype)

        tensor_h = torch.tensor(
            config.init_h_shape, dtype=dtype_map[config.init_h_dtype])
        init_h = torch.nn.init.constant_(tensor=tensor_h, val=0.0)

        tensor_c = torch.tensor(
            config.init_c_shape, dtype=dtype_map[config.init_c_dtype])
        init_c = torch.nn.init.constant_(tensor=tensor_c, val=0.0)

        rnn = torch.nn.LSTM(
            input_size=config.input_shape[-1],
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=0.2,
            bidirectional=config.is_bidirec)
        # move model and input to cuda device
        if torch.cuda.is_available():
            rnn.cuda()
            input = input.cuda()

        # rnn_out, (last_h, last_c) = rnn(input, (init_h, init_c))
        # If (h_0, c_0) is not provided, both h_0 and c_0 default to zero
        rnn_out, (last_h, last_c) = rnn(input)

        self.feed_list = [input]
        self.fetch_list = [rnn_out]


if __name__ == '__main__':
    test_main(pd_dy_obj=PDLstm(), torch_obj=TorchLstm(), config=LstmConfig())
