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


class RnnConfig(APIConfig):
    def __init__(self):
        super(RnnConfig, self).__init__("rnn")

    def to_pytorch(self):
        torch_config = super(RnnConfig, self).to_pytorch()
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool
        }
        torch_config.init_h_dtype = dtype_map[self.init_h_dtype]
        return torch_config


class PDRnn(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        input = self.variable(
            name="input", shape=config.input_shape, dtype=config.input_dtype)
        # The shape of init_h is [batch_size, hidden_size],which is referenced from
        # https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/nn/layer/rnn/SimpleRNNCell_cn.html#simplernncell
        init_h = paddle.full(
            shape=[config.init_h_shape[1], config.hidden_size],
            dtype=config.init_h_dtype,
            fill_value=0.0)

        cell = paddle.nn.SimpleRNNCell(config.input_shape[-1],
                                       config.hidden_size)

        rnn = paddle.nn.RNN(cell)
        output, last_h = rnn(input, init_h)

        self.feed_list = [input]
        self.fetch_list = [output]


class TorchRnn(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        config.input_shape = [
            config.input_shape[1], config.input_shape[0], config.input_shape[2]
        ]
        input = self.variable(
            name="input", shape=config.input_shape, dtype=config.input_dtype)
        tensor_h = torch.empty(config.init_h_shape, dtype=config.init_h_dtype)
        init_h = torch.nn.init.constant_(tensor=tensor_h, val=0.0)

        rnn = torch.nn.RNN(config.input_shape[-1], config.hidden_size,
                           config.num_layers)

        # Put input and hidden layer into the same device
        if torch.cuda.is_available():
            rnn = rnn.cuda()
            input = input.cuda()
            init_h = init_h.cuda()

        rnn_out, last_h = rnn(input, init_h)

        self.feed_list = [input]
        self.fetch_list = [rnn_out]


if __name__ == "__main__":
    test_main(pd_dy_obj=PDRnn(), torch_obj=TorchRnn(), config=RnnConfig())
