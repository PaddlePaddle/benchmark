# copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import print_function

import numpy as np
import paddle.fluid as fluid

import sys
sys.path.append("..")
from common import paddle_api_benchmark as paddle_api
from common import tensorflow_api_benchmark as tensorflow_api


def feed_var(spec):
    if not isinstance(spec, dict):
        raise TypeError("Expected spec a dict, received a ", type(spec))

    assert spec.get("shape", None) is not None
    assert spec.get("dtype", None) is not None

    shape = spec["shape"]
    dtype = spec["dtype"]
    range = None
    if spec.get("range", None) is not None:
        range = spec["range"]
        if not isinstance(range, tuple) and not isinstance(range, list):
            TypeError("Expected range a tuple or a list, received a ", type(range))
        assert len(range) == 2

    if spec.get("data", None) is not None:
        data = spec["data"]
    else:
        if dtype == "int64" or dtype == "int32":
            assert range is not None
            data = np.random.randint(range[0], range[1], shape).astype(dtype)
        else:
            data = np.random.random(shape).astype(dtype)
            if range is not None:
                data = range[0] + (range[1] - range[0]) * data
    return data


def feed_paddle(obj, feed_spec=None):
    if feed_spec is not None:
        if not isinstance(feed_spec, list):
            feed_spec = [feed_spec]
        assert len(obj.feed_vars) == len(feed_spec)

    feed_list = []
    for i in range(len(obj.feed_vars)):
        var = obj.feed_vars[i]
        if var.type != fluid.core.VarDesc.VarType.LOD_TENSOR:
            raise TypeError("Feed data of non LoDTensor is not supported.")

        if feed_spec is not None:
            spec = feed_spec[i]
        else:
            spec = {}
            
        if spec.get("shape", None) is None:
            spec["shape"] = var.shape
        if spec.get("dtype", None) is None:
            spec["dtype"] = paddle_api.convert_dtype(var.dtype)

        data = feed_var(spec)
        feed_list.append(data)
    return feed_list


def feed_tensorflow(obj, feed_list=None, feed_spec=None):
    if feed_list is not None:
        assert len(obj.feed_list) == len(feed_list)

        for i in range(len(obj.feed_list)):
            var = obj.feed_list[i]

            if feed_spec is not None:
                spec = feed_spec[i]
                if spec.get("permute", None) is not None:
                    feed_list[i] = np.transpose(feed_list[i], spec["permute"]) 

            assert var.shape == feed_list[i].shape
            assert tensorflow_api.convert_dtype(var.dtype, to_string=False) == feed_list[i].dtype
    else:
        feed_list = []
        for i in range(len(obj.feed_list)):
            var = obj.feed_list[i]

            if feed_spec is not None:
                spec = feed_spec[i]
            else:
                spec = {}
        
            if spec.get("shape", None) is None:
                spec["shape"] = var.shape
            if spec.get("dtype", None) is None:
                spec["dtype"] = tensorflow_api.convert_dtype(var.dtype)

            data = feed_var(spec)
            feed_list.append(data)
    return feed_list
