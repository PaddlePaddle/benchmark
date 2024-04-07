#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from common_import import *


def get_sin_cos_tensor(seq_len, head_dim, sign=1):
    pos_seq = np.arange(0, seq_len, 1, dtype="float32")
    indices = np.arange(0, head_dim, 2, dtype="float32")

    indices = 1 / 10000**(indices / head_dim)
    sinusoid_inp = np.expand_dims(pos_seq, axis=1) * np.expand_dims(indices,
                                                                    axis=0)

    sin_sin = np.empty((seq_len * head_dim), dtype=np.float32)
    cos_cos = np.empty((seq_len * head_dim), dtype=np.float32)
    numpy_array = sinusoid_inp
    iter_array = np.nditer(numpy_array)

    i = 0

    for value in iter_array:
        sin_sin[i * 2] = sign * np.sin(value)
        cos_cos[i * 2 + 0] = np.cos(value)
        sin_sin[i * 2 + 1] = np.sin(value)
        cos_cos[i * 2 + 1] = np.cos(value)
        i += 1

    return np.reshape(sin_sin, [1, seq_len, 1, head_dim]), np.reshape(
        cos_cos, [1, seq_len, 1, head_dim])


@benchmark_registry.register("fused_rotary_position_embedding")
class FusedRoPEConfig(APIConfig):

    def __init__(self):
        super(FusedRoPEConfig,
              self).__init__("fused_rotary_position_embedding")
        self.run_torch = False

    def init_from_json(self, filename, config_id=0, unknown_dim=16):
        super(FusedRoPEConfig, self).init_from_json(filename, config_id,
                                                    unknown_dim)
        # check whether q,k,v are set None
        self.has_q = True if hasattr(self, "q_shape") else False
        self.has_k = True if hasattr(self, "k_shape") else False
        self.has_v = True if hasattr(self, "v_shape") else False

        if self.cached_sin_cos:
            seq_len = self.q_shape[1]
            head_dim = self.q_shape[3]
            self.sin_value, self.cos_value = get_sin_cos_tensor(
                seq_len, head_dim, 1)

            self.sin_shape, self.cos_shape = self.sin_value.shape, self.cos_value.shape
            self.sin_dtype, self.cos_dtype = "float32", "float32"
        else:
            self.sin_value, self.cos_value = None, None


@benchmark_registry.register("fused_rotary_position_embedding")
class PaddleFusedRoPE(PaddleOpBenchmarkBase):

    def __init__(self, *args, **kwargs):
        super(PaddleFusedRoPE, self).__init__(*args, **kwargs)
        self.sin_tensor = None
        self.cos_tensor = None

    def build_graph(self, config):
        q = self.variable(name='q', shape=config.q_shape,
                          dtype=config.q_dtype) if config.has_q else None
        k = self.variable(name='k', shape=config.k_shape,
                          dtype=config.k_dtype) if config.has_k else None
        v = self.variable(name='v', shape=config.v_shape,
                          dtype=config.v_dtype) if config.has_v else None

        sin = None if config.sin_value is None else self.variable(
            name='sin',
            shape=config.sin_shape,
            dtype=config.sin_dtype,
            value=config.sin_value)
        cos = None if config.cos_value is None else self.variable(
            name='cos',
            shape=config.cos_shape,
            dtype=config.cos_dtype,
            value=config.cos_value)

        results = paddle.incubate.nn.functional.fused_rotary_position_embedding(
            q=q,
            k=k,
            v=v,
            sin=sin,
            cos=cos,
            position_ids=config.position_ids,
            use_neox_rotary_style=config.use_neox_rotary_style)

        self.feed_list = []
        self.fetch_list = []
        for idx, item in enumerate([q, k, v]):
            if item is not None:
                self.feed_list.append(item)
                self.feed_list.append(results[idx])
