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

from main import test_main, test_main_without_json

import sys
sys.path.append("..")
from common.paddle_api_benchmark import PaddleAPIBenchmarkBase
from common.tensorflow_api_benchmark import TensorflowAPIBenchmarkBase
from common.api_param import APIConfig

try:
    import paddle.fluid as fluid
except ImportError:
    sys.stderr.write(
        "Cannot import paddle.fluid, maybe paddle is not installed.\n")

try:
    import tensorflow as tf
except ImportError:
    sys.stderr.write(
        "Cannot import tensorflow, maybe tensorflow is not installed.\n")
