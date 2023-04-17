#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
from paddle import _legacy_C_ops


# no dropout_nd in pytorch
def dropout_nd(x,
               p=0.5,
               axis=None,
               training=True,
               mode="upscale_in_train",
               name=None):
    drop_axes = [axis] if isinstance(axis, int) else list(axis)
    seed = None
    mode = ('downgrade_in_infer'
            if mode == 'downscale_in_infer' else mode)  # semantic transfer
    out = _legacy_C_ops.dropout_nd(
        x,
        'dropout_prob',
        p,
        'is_test',
        not training,
        'fix_seed',
        seed is not None,
        'seed',
        seed if seed is not None else 0,
        'dropout_implementation',
        mode,
        'axis',
        drop_axes, )
    return out


@benchmark_registry.register("dropout_nd")
class PaddleDropout(PaddleOpBenchmarkBase):
    def build_graph(self, config):
        x = self.variable(name='x', shape=config.x_shape, dtype=config.x_dtype)
        result = dropout_nd(
            x=x, p=config.p, axis=config.axis, mode=config.mode)

        self.feed_list = [x]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result[0], self.feed_list)
