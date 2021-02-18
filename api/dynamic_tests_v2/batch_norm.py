#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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


class BatchNormConfig(APIConfig):
    def __init__(self):
        super(BatchNormConfig, self).__init__('batch_norm')

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(BatchNormConfig, self).init_from_json(filename, config_id,
                                                    unknown_dim)
        if len(self.x_shape) == 4:
            if self.data_format == "NCHW":
                self.num_channels = self.x_shape[1]
            else:
                self.num_channels = self.x_shape[3]
        else:
            self.num_channels = self.x_shape[1]


class PDBatchNorm(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name='weight', shape=[config.num_channels], dtype=config.x_dtype)
        bias = self.variable(
            name='bias', shape=[config.num_channels], dtype=config.x_dtype)

        if not hasattr(self, "running_mean") or self.running_mean is None:
            self.running_mean = paddle.create_parameter(
                name='running_mean',
                shape=[config.num_channels],
                dtype=config.x_dtype,
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(0.5)))
            self.running_mean.stop_gradient = True
        if not hasattr(self, "running_var") or self.running_var is None:
            self.running_var = paddle.create_parameter(
                name='running_var',
                shape=[config.num_channels],
                dtype=config.x_dtype,
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(0.1)))
            self.running_var.stop_gradient = True

        result = paddle.nn.functional.batch_norm(
            x=x,
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=weight,
            bias=bias,
            epsilon=config.epsilon,
            momentum=config.momentum,
            training=config.training,
            data_format=config.data_format)

        self.feed_list = [x, weight, bias]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, weight, bias])


class TorchBatchNorm(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name='weight', shape=[config.num_channels], dtype=config.x_dtype)
        bias = self.variable(
            name='bias', shape=[config.num_channels], dtype=config.x_dtype)

        if not hasattr(self, "running_mean") or self.running_mean is None:
            self.running_mean = torch.zeros([config.num_channels]).cuda()
        if not hasattr(self, "running_var") and self.running_var is None:
            self.running_var = torch.ones([config.num_channels]).cuda()

        result = torch.nn.functional.batch_norm(
            input=x,
            running_mean=self.running_mean,
            running_var=self.running_var,
            weight=weight,
            bias=bias,
            training=config.training,
            momentum=config.momentum,
            eps=config.epsilon)

        self.feed_list = [x, weight, bias]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, weight, bias])


if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDBatchNorm(),
        torch_obj=TorchBatchNorm(),
        config=BatchNormConfig())
