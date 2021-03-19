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


class SyncBatchNormConfig(APIConfig):
    def __init__(self):
        super(SyncBatchNormConfig, self).__init__('sync_batch_norm')

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(SyncBatchNormConfig, self).init_from_json(filename, config_id,
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
        super(SyncBatchNormConfig, self).convert_to_fp16()
        if self.data_format == "NHWC":
            paddle.fluid.set_flags({
                'FLAGS_cudnn_SyncBatchNorm_spatial_persistent': 1
            })
        self._set_param_dtype()


class PDSyncBatchNorm(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        weight_attr = paddle.ParamAttr(
            learning_rate=0.5,
            regularizer=paddle.regularizer.L2Decay(1.0),
            trainable=True)

        bias_attr = paddle.ParamAttr(
            learning_rate=0.5,
            regularizer=paddle.regularizer.L2Decay(1.0),
            trainable=True)

        sync_batch_norm = paddle.nn.SyncBatchNorm(
            num_features=config.num_channels,
            epsilon=config.epsilon,
            momentum=config.momentum,
            data_format=config.data_format)
        result = sync_batch_norm(x)
        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TorchSyncBatchNorm(PytorchAPIBenchmarkBase):
    def build_graph(self, config):
        print("sorry pytorch is unsupported!")


#        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
#        device = torch.device("cuda")
#        sync_batch_norm = torch.nn.SyncBatchNorm(
#            num_features = config.num_channels,
#            eps = config.epsilon,
#            momentum=config.momentum,
#            affine=True,
#            track_running_stats = config.training,
#            process_group = process_group)
#        result = sync_batch_norm(x)
#        self.feed_list = [x]
#        self.fetch_list = [result]
#        if config.backward:
#            self.append_gradients(result, [x])

if __name__ == '__main__':
    test_main(
        pd_dy_obj=PDSyncBatchNorm(),
        torch_obj=TorchSyncBatchNorm(),
        config=SyncBatchNormConfig())
