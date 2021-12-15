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
        # num_channels
        if len(self.x_shape) == 4:
            self.num_channels = self.x_shape[
                1] if self.data_format == "NCHW" else self.x_shape[3]
        else:
            self.num_channels = self.x_shape[1]

        self._set_param_dtype()
        if self.data_format == 'NHWC':
            print(
                "Warning:\n"
                "  1. PyTorch does not have data_format param, it only support NHWC format.\n"
            )
            self.run_torch = False

    def _set_param_dtype(self):
        # dtype of parameters
        self.param_dtype = "float32" if self.x_dtype == "float16" else self.x_dtype

    def convert_to_fp16(self):
        super(BatchNormConfig, self).convert_to_fp16()
        if self.data_format == "NHWC":
            paddle.fluid.set_flags({
                'FLAGS_cudnn_batchnorm_spatial_persistent': 1
            })
        self._set_param_dtype()


class PDBatchNorm(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        weight = self.variable(
            name='weight',
            shape=[config.num_channels],
            dtype=config.param_dtype)
        bias = self.variable(
            name='bias', shape=[config.num_channels], dtype=config.param_dtype)

        # running_mean and running_var will be used as input and output.
        # When training and use_global_stats is False, input mean and variance
        # will not be used, output mean and variance_out will be updated.
        # So it is not need to initialize running_mean and running_variance
        # every time for this case.
        if not config.training or not hasattr(self, "_running_mean"):
            self._running_mean = paddle.create_parameter(
                name='running_mean',
                shape=[config.num_channels],
                dtype=config.param_dtype,
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(0)))
            self._running_mean.stop_gradient = True
        running_mean = self._running_mean

        if not config.training or not hasattr(self, "_running_var"):
            self._running_var = paddle.create_parameter(
                name='running_var',
                shape=[config.num_channels],
                dtype=config.param_dtype,
                attr=paddle.ParamAttr(
                    initializer=paddle.nn.initializer.Constant(1)))
            self._running_var.stop_gradient = True
        running_var = self._running_var

        result = paddle.nn.functional.batch_norm(
            x=x,
            running_mean=running_mean,
            running_var=running_var,
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
            name='weight',
            shape=[config.num_channels],
            dtype=config.param_dtype)
        bias = self.variable(
            name='bias', shape=[config.num_channels], dtype=config.param_dtype)

        # running_mean and running_var will be used as input and output.
        # When training and use_global_stats is False, input mean and variance
        # will not be used, output mean and variance_out will be updated.
        # So it is not need to initialize running_mean and running_variance
        # every time for this case.
        device = torch.device("cuda" if use_gpu() and torch.cuda.is_available()
                              else "cpu")
        if not config.training or not hasattr(self, "_running_mean"):
            self._running_mean = torch.zeros(
                [config.num_channels], device=device)
        running_mean = self._running_mean

        if not config.training or not hasattr(self, "_running_var"):
            self._running_var = torch.ones(
                [config.num_channels], device=device)
        running_var = self._running_var

        result = torch.nn.functional.batch_norm(
            input=x,
            running_mean=running_mean,
            running_var=running_var,
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
