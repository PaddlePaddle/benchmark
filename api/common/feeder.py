# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import paddle_api_benchmark as paddle_api
import tensorflow_api_benchmark as tensorflow_api


def copy_feed_spec(feed_spec):
    if feed_spec is None:
        return None
    if not isinstance(feed_spec, list):
        feed_spec = [feed_spec]

    copy = []
    for feed_item in feed_spec:
        item = {}
        for key, value in feed_item.items():
            item[key] = value
        copy.append(item)
    return copy


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
            TypeError("Expected range a tuple or a list, received a ",
                      type(range))
        assert len(range) == 2

    if spec.get("data", None) is not None:
        data = spec["data"]
    else:
        if dtype == "int64" or dtype == "int32":
            data = np.random.randint(100, size=shape, dtype=dtype)
            if range is not None:
                data = np.random.randint(
                    range[0], range[1], size=shape, dtype=dtype)
        if dtype == "bool":
            data = np.random.randint(2, size=shape, dtype=bool)
        else:
            data = np.random.random(shape).astype(dtype)
            if range is not None:
                data = range[0] + (range[1] - range[0]) * data
    return data


def feed_paddle(obj, feed_spec=None):
    feed_spec = copy_feed_spec(feed_spec)
    assert isinstance(feed_spec, list)
    assert len(obj.feed_vars) == len(
        feed_spec
    ), "Expected the number of feeding vars ({}) to be equal to the length of feed_spec ({}).".format(
        len(obj.feed_vars), len(feed_spec))

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


def check_shape(shape, shape_ref):
    if shape == shape_ref:
        return True
    if shape + [1] == shape_ref or shape == shape_ref + [1]:
        return True
    return False


def feed_tensorflow(obj, feed_list=None, feed_spec=None):
    if feed_spec is not None:
        if not isinstance(feed_spec, list):
            feed_spec = [feed_spec]
        assert len(obj.feed_list) == len(feed_spec)

    if feed_list is not None:
        assert len(obj.feed_list) == len(feed_list)

        for i in range(len(obj.feed_list)):
            var = obj.feed_list[i]

            if feed_spec is not None:
                spec = feed_spec[i]
                if spec.get("permute", None) is not None:
                    feed_list[i] = np.transpose(feed_list[i], spec["permute"])

            assert check_shape(var.shape, feed_list[i].shape)
            feed_list[i] = feed_list[i].reshape(var.shape)

            dtype = tensorflow_api.convert_dtype(var.dtype, to_string=False)
            if dtype != feed_list[i].dtype:
                feed_list[i] = feed_list[i].astype(dtype)
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
