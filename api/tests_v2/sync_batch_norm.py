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
        print("Warning:\n" "sorry tensorflow is unsupported!\n")
        self.run_tf = False

        # num_channels
        if len(self.x_shape) == 4:
            if self.data_format == "NCHW":
                self.num_channels = self.x_shape[1]
            else:
                self.num_channels = self.x_shape[3]
        else:
            self.num_channels = self.x_shape[1]


class PDSyncBatchNorm(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)

        sync_batch_norm = paddle.nn.SyncBatchNorm(
            num_features=config.num_channels,
            epsilon=config.epsilon,
            momentum=config.momentum,
            data_format=config.data_format)

        result = sync_batch_norm(x)
        self.feed_vars = [x]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x])


if __name__ == '__main__':
    test_main(PDSyncBatchNorm(), config=SyncBatchNormConfig())
