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
    sys.stderr.write(
        "Cannot import paddle or paddle.fluid, maybe paddle is not installed.\n"
    )

try:
    import tensorflow as tf
except ImportError:
    sys.stderr.write(
        "Cannot import tensorflow, maybe tensorflow is not installed.\n")

package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_path)

from common.paddle_api_benchmark import PaddleAPIBenchmarkBase
from common.tensorflow_api_benchmark import TensorflowAPIBenchmarkBase
from common.api_param import APIConfig
from common.main import test_main, test_main_without_json


def unsqueeze_short(short, long):
    """
    Unsqueeze the short shape to the same length of the long's.
    For example: short is [16, 2048] and long is [16, 2048, 7, 7],
    it will return [16, 2048, 1, 1].
    """
    short_extend = np.ones([len(long)], dtype=np.int32).tolist()
    start = 0
    for value in short:
        for i in range(start, len(long)):
            if long[i] == value:
                short_extend[i] = value
                start = i
                break
    return short_extend
