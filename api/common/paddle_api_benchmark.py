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

import abc, six
import logging
import warnings
import numpy as np
import sys

from common.paddle_op_benchmark import PaddleOpBenchmarkBase

try:
    import paddle
    import paddle.fluid as fluid
except Exception as e:
    sys.stderr.write(
        "Cannot import paddle.fluid, maybe paddle is not installed.\n")

try:
    paddle.enable_static()
except Exception:
    print(
        "The paddle version is less than 2.0, it can not use paddle.enable_static()"
    )


def convert_dtype(dtype, to_string=True):
    def _trans(to_string, dtype_str, np_dtype):
        dtype = dtype_str if to_string else np.dtype(np_dtype)
        return dtype

    if not isinstance(dtype, fluid.core.VarDesc.VarType):
        raise TypeError("dtype is not of type fluid.core.VarDesc.VarType")
    if dtype == fluid.core.VarDesc.VarType.FP32:
        return _trans(to_string, "float32", np.float32)
    elif dtype == fluid.core.VarDesc.VarType.FP64:
        return _trans(to_string, "float64", np.float64)
    elif dtype == fluid.core.VarDesc.VarType.FP16:
        return _trans(to_string, "float16", np.float16)
    elif dtype == fluid.core.VarDesc.VarType.INT32:
        return _trans(to_string, "int32", np.int32)
    elif dtype == fluid.core.VarDesc.VarType.INT16:
        return _trans(to_string, "int16", np.int16)
    elif dtype == fluid.core.VarDesc.VarType.INT64:
        return _trans(to_string, "int64", np.int64)
    elif dtype == fluid.core.VarDesc.VarType.BOOL:
        return _trans(to_string, "bool", np.bool)
    elif dtype == fluid.core.VarDesc.VarType.INT16:
        return _trans(to_string, "uint16", np.uint16)
    elif dtype == fluid.core.VarDesc.VarType.UINT8:
        return _trans(to_string, "uint8", np.uint8)
    elif dtype == fluid.core.VarDesc.VarType.INT8:
        return _trans(to_string, "int8", np.int8)
    else:
        raise ValueError("Unsupported dtype %s" % dtype)


@six.add_metaclass(abc.ABCMeta)
class PaddleAPIBenchmarkBase(PaddleOpBenchmarkBase):
    def __init__(self):
        super(PaddleAPIBenchmarkBase, self).__init__("static")
        self.scope = None
        self.feed_vars = None
        self.fetch_vars = None

    @abc.abstractmethod
    def build_program(self, config=None):
        pass

    def build_graph(self, config=None):
        self.build_program(config)
        self.feed_list = self.feed_vars
        self.fetch_list = self.fetch_vars
