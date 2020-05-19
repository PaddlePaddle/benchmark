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

import collections
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


def check_shape_and_dtype(shape, dtype, value):
    assert list(shape) == list(value.shape) or list(shape) + [
        1
    ] == list(value.shape) or list(shape) == list(value.shape) + [1]
    value = value.reshape(shape)

    # Allow different data type
    if dtype != value.dtype:
        value = value.astype(dtype)

    return value


def generate_random_data(shape, dtype, range=None, value=None):
    if range is not None:
        if not isinstance(range, tuple) and not isinstance(range, list):
            raise TypeError("Expected range a tuple or a list, received a ",
                            type(range))
        assert len(range) == 2

    if value is not None:
        assert isinstance(value, np.ndarray)
        data = check_shape_and_dtype(shape, dtype, value)
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


def generate_numpy_data(spec):
    if not isinstance(spec, dict):
        raise TypeError("Expected spec a dict, received a ", type(spec))

    assert spec.get("shape", None) is not None
    assert spec.get("dtype", None) is not None

    shape = spec["shape"]
    dtype = spec["dtype"]
    range = spec.get("range", None)
    value = spec.get("value", None)
    return generate_random_data(shape, dtype, range, value)


def feed_paddle(feed_vars, feed_spec=None):
    assert isinstance(feed_vars, list)
    if feed_spec is not None:
        if not isinstance(feed_spec, list):
            feed_spec = [feed_spec]
        assert len(feed_vars) == len(
            feed_spec
        ), "Expected the number of feeding vars (%d) to be equal to the length of feed_spec (%d)." % (
            len(feed_vars), len(feed_spec))

    feed_dict = collections.OrderedDict()
    for i in range(len(feed_vars)):
        var = feed_vars[i]
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

        feed_dict[var.name] = generate_numpy_data(spec)
    return feed_dict


def feed_tensorflow(feed_list, feed_dict_paddle=None, feed_spec=None):
    feed_spec = copy_feed_spec(feed_spec)

    feed_dict_tensorflow = collections.OrderedDict()
    if feed_dict_paddle is not None:
        for i in range(len(feed_dict_paddle)):
            item = feed_dict_paddle.items()[i]
            name = item[0]
            value = item[1]

            if feed_spec is not None:
                spec = feed_spec[i]
                if spec.get("permute", None) is not None:
                    value = np.transpose(value, spec["permute"])

            if feed_list is not None:
                var = feed_list[i]
                value = check_shape_and_dtype(var.shape, var.dtype, value)
                feed_dict_tensorflow[var.name] = value
            else:
                feed_dict_tensorflow[name] = value
    else:
        assert feed_list is not None
        for i in range(len(feed_list)):
            var = feed_list[i]

            if feed_spec is not None:
                spec = feed_spec[i]
            else:
                spec = {}

            if spec.get("shape", None) is None:
                spec["shape"] = var.shape
            if spec.get("dtype", None) is None:
                spec["dtype"] = tensorflow_api.convert_dtype(var.dtype)

            feed_dict_tensorflow[var.name] = generate_numpy_data(spec)
    return feed_dict_tensorflow
