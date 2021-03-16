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


class GroupNormConfig(APIConfig):
    def __init__(self):
        super(GroupNormConfig, self).__init__('group_norm')
        self.run_tf = False

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(GroupNormConfig, self).init_from_json(filename, config_id,
                                                    unknown_dim)
        # num_channels
        if len(self.x_shape) == 4:
            self.num_channels = self.x_shape[1] \
                if self.data_format == "NCHW" else self.x_shape[3]
        else:
            self.num_channels = self.x_shape[1]

    def to_tensorflow(self):
        tf_config = super(GroupNormConfig, self).to_tensorflow()
        # axis
        if len(self.x_shape) == 4:
            tf_config.channel_axis = -3 if self.data_format == "NCHW" else -1
        return tf_config


class PDGroupNorm(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        paddle_group_norm = paddle.nn.GroupNorm(
            num_channels=config.num_channels,
            num_groups=config.num_groups,
            epsilon=config.epsilon)
        result = paddle_group_norm(x)

        self.feed_vars = [x]
        self.fetch_vars = [result]
        if config.backward:
            self.append_gradients(result, [x])


class TFGroupNorm(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)


'''
# Execute condition: tensorflow 2.0 with tensorflow_addons API
        tfa_group_norm = tfa.layers.GroupNormalization(
            groups=config.num_groups,
            axis=config.channel_axis,
            epsilon=config.epsilon)
        result = tfa_group_norm(x)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x])
'''

if __name__ == '__main__':
    test_main(PDGroupNorm(), TFGroupNorm(), config=GroupNormConfig())
