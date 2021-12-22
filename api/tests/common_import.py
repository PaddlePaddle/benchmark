#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os, sys
import numpy as np

try:
    import paddle
except ImportError:
    sys.stderr.write("Cannot import paddle, maybe paddle is not installed.\n")

try:
    import torch
except ImportError:
    sys.stderr.write("Cannot import pytorch, maybe paddle is not installed.\n")

package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_path)

from common.paddle_dynamic_api_benchmark import PaddleDynamicAPIBenchmarkBase
from common.pytorch_api_benchmark import PytorchAPIBenchmarkBase
from common.api_param import APIConfig
from common.registry import benchmark_registry


def use_gpu():
    return os.environ.get("CUDA_VISIBLE_DEVICES", None) != ""


def unsqueeze_short(short, long):
    """
    Unsqueeze the short shape to the same length of the long's.
    For example: short is [16, 2048] and long is [16, 2048, 7, 7],
    it will return [16, 2048, 1, 1].
    """
    # Extend short with 0s.
    short_extend_zeros = np.zeros([len(long)], dtype=np.int32).tolist()
    start = 0
    for value in short:
        for i in range(start, len(long)):
            if long[i] == value:
                short_extend_zeros[i] = value
                start = i
                break
    # Remove the 0s on the front and change 0s on the middle to 1s, [0, M, 0, N] -> [M, 1, N]
    short_extend = []
    first_nonzero_idx = -1
    for i in range(len(short_extend_zeros)):
        if first_nonzero_idx == -1 and short_extend_zeros[i] != 0:
            first_nonzero_idx = i
        if first_nonzero_idx > -1:
            if short_extend_zeros[i] == 0:
                short_extend.append(1)
            else:
                short_extend.append(short_extend_zeros[i])
    return short_extend


def numel(shape):
    assert isinstance(
        shape, list), "Expect shape to be a list, but recieved {}".format(
            type(shape))
    return np.prod(np.array(shape))


def sizeof(dtype):
    assert isinstance(
        dtype, str), "Expect dtype to be a string, but recieved {}".format(
            type(dtype))
    if dtype in ["float64", "double", "int64", "long"]:
        return 8
    elif dtype in ["float32", "float", "int32", "int"]:
        return 4
    elif dtype in ["float16", "bfloat16"]:
        return 2
    elif dtype in ["bool"]:
        return 1
    else:
        raise ValueError("{} is not supported.".format(dtype))
