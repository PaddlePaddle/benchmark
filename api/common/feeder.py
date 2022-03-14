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

import collections
import numpy as np


def _convert_paddle_dtype(dtype, to_string=True):
    import paddle

    def _trans(to_string, dtype_str, np_dtype):
        dtype = dtype_str if to_string else np.dtype(np_dtype)
        return dtype

    if not isinstance(dtype, paddle.fluid.core.VarDesc.VarType):
        raise TypeError("dtype is not of type fluid.core.VarDesc.VarType")
    if dtype == paddle.fluid.core.VarDesc.VarType.FP32:
        return _trans(to_string, "float32", np.float32)
    elif dtype == paddle.fluid.core.VarDesc.VarType.FP64:
        return _trans(to_string, "float64", np.float64)
    elif dtype == paddle.fluid.core.VarDesc.VarType.FP16:
        return _trans(to_string, "float16", np.float16)
    elif dtype == paddle.fluid.core.VarDesc.VarType.INT32:
        return _trans(to_string, "int32", np.int32)
    elif dtype == paddle.fluid.core.VarDesc.VarType.INT16:
        return _trans(to_string, "int16", np.int16)
    elif dtype == paddle.fluid.core.VarDesc.VarType.INT64:
        return _trans(to_string, "int64", np.int64)
    elif dtype == paddle.fluid.core.VarDesc.VarType.BOOL:
        return _trans(to_string, "bool", np.bool)
    elif dtype == paddle.fluid.core.VarDesc.VarType.INT16:
        return _trans(to_string, "uint16", np.uint16)
    elif dtype == paddle.fluid.core.VarDesc.VarType.UINT8:
        return _trans(to_string, "uint8", np.uint8)
    elif dtype == paddle.fluid.core.VarDesc.VarType.INT8:
        return _trans(to_string, "int8", np.int8)
    else:
        raise ValueError("Unsupported dtype %s" % dtype)


def _convert_tensorflow_dtype(dtype, to_string=True):
    import tensorflow as tf

    def _trans(to_string, dtype_str, np_dtype):
        dtype = dtype_str if to_string else np.dtype(np_dtype)
        return dtype

    if dtype == tf.float16:
        # tf.float16: 16-bit half-precision floating-point.
        return _trans(to_string, "float16", np.float16)
    elif dtype == tf.float32:
        # tf.float32: 32-bit single-precision floating-point.
        return _trans(to_string, "float32", np.float32)
    elif dtype == tf.float64:
        # tf.float64: 64-bit double-precision floating-point.
        return _trans(to_string, "float64", np.float64)
    elif dtype == tf.int8:
        # tf.int8: 8-bit signed integer.
        return _trans(to_string, "int8", np.int8)
    elif dtype == tf.uint8:
        # tf.uint8: 8-bit unsigned integer.
        return _trans(to_string, "uint8", np.uint8)
    elif dtype == tf.uint16:
        # tf.uint16: 16-bit unsigned integer.
        return _trans(to_string, "uint16", np.uint16)
    elif dtype == tf.uint32:
        # tf.uint32: 32-bit unsigned integer.
        return _trans(to_string, "uint32", np.uint32)
    elif dtype == tf.uint64:
        # tf.uint64: 64-bit unsigned integer.
        return _trans(to_string, "uint64", np.uint64)
    elif dtype == tf.int16:
        # tf.int16: 16-bit signed integer.
        return _trans(to_string, "int16", np.int16)
    elif dtype == tf.int32:
        # tf.int32: 32-bit signed integer.
        return _trans(to_string, "int32", np.int32)
    elif dtype == tf.int64:
        # tf.int64: 64-bit signed integer.
        return _trans(to_string, "int64", np.int64)
    elif dtype == tf.bool:
        # tf.bool: Boolean.
        return _trans(to_string, "bool", np.bool)
    else:
        # tf.bfloat16: 16-bit truncated floating-point.
        # tf.complex64: 64-bit single-precision complex.
        # tf.complex128: 128-bit double-precision complex.
        # tf.string: String.
        # tf.qint8: Quantized 8-bit signed integer.
        # tf.quint8: Quantized 8-bit unsigned integer.
        # tf.qint16: Quantized 16-bit signed integer.
        # tf.quint16: Quantized 16-bit unsigned integer.
        # tf.qint32: Quantized 32-bit signed integer.
        # tf.resource: Handle to a mutable resource.
        # tf.variant: Values of arbitrary types.
        raise ValueError("Unsupported dtype %s" % dtype)


def copy_feed_spec(feed_spec):
    if feed_spec is None:
        return None
    if not isinstance(feed_spec, list):
        feed_spec = [feed_spec]

    copy = []
    for feed_item in feed_spec:
        assert isinstance(feed_item, dict)
        item_copy = {}
        for key, value in feed_item.items():
            item_copy[key] = value
        copy.append(item_copy)
    return copy


def check_shape_and_dtype(shape, dtype, value):
    assert list(shape) == list(value.shape) or list(shape) + [
        1
    ] == list(value.shape) or list(shape) == list(
        value.shape) + [1], "Expected shape: %s. Recieved shape: %s." % (
            shape, value.shape)
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
        if isinstance(value, list):
            value = np.array(value)
        assert isinstance(
            value, np.ndarray
        ), "Expected value's type to be numpy.ndarray, but recieved {}.".format(
            type(value))
        data = check_shape_and_dtype(shape, dtype, value)
    else:
        if dtype == "int64" or dtype == "int32":
            data = np.random.randint(100, size=shape, dtype=dtype)
            if range is not None:
                data = np.random.randint(
                    range[0], range[1], size=shape, dtype=dtype)
        elif dtype == "bool":
            data = np.random.randint(2, size=shape, dtype=bool)
        elif dtype == "uint8" or dtype == "uint16":
            data = np.random.randint(0, 100, size=shape, dtype=dtype)
            range_low = max(0, range[0])
            if range is not None:
                data = np.random.randint(
                    range_low, range[1], size=shape, dtype=dtype)
        else:
            data = np.random.random(shape).astype(dtype)
            if range is not None:
                data = range[0] + (range[1] - range[0]) * data
    return data


class FeederAdapter(object):
    def __init__(self, framework, feed_spec, feed_list):
        assert framework in ["paddle", "tensorflow", "pytorch"]
        if feed_spec is not None:
            assert len(feed_list) == len(
                feed_spec
            ), "Expected the number of feeding vars (%d) to be equal to the length of feed_spec (%d)." % (
                len(feed_list), len(feed_spec))

        self.__framework = framework
        self.__feed_spec = copy_feed_spec(feed_spec)
        self.__feed_list = feed_list

    def to_paddle(self, feed_vars=None):
        import paddle.fluid as fluid

        if self.__framework == "paddle":
            return self.__feed_list
        else:
            if feed_vars is not None:
                assert isinstance(feed_vars, list)
                assert len(feed_vars) == len(self.__feed_list)

            feed_list = []
            for i in range(len(self.__feed_list)):
                value = self.__feed_list[i]

                if self.__feed_spec is not None and self.__feed_spec[i].get(
                        "permute", None) is not None:
                    permute_paddle2other = self.__feed_spec[i]["permute"]
                    permute_other2paddle = [0] * len(permute_paddle2other)
                    for pos in range(len(permute_paddle2other)):
                        permute_other2paddle[permute_paddle2other[pos]] = pos
                    value = np.transpose(value, permute_other2paddle)

                # On dynamic mode, the check is skipped.
                if feed_vars is not None:
                    var = feed_vars[i]
                    if var.type != fluid.core.VarDesc.VarType.LOD_TENSOR:
                        raise TypeError(
                            "Feed data of non LoDTensor is not supported.")

                    # Check shape and dtype
                    var_shape = var.shape
                    var_dtype = _convert_paddle_dtype(
                        var.dtype, to_string=True)
                    value = check_shape_and_dtype(var_shape, var_dtype, value)

                feed_list.append(value)
            return feed_list

    def to_tensorflow(self, feed_vars=None):
        target_framework = "tensorflow"
        if self.__framework == target_framework:
            return self.__feed_list
        else:
            return self._to_other(target_framework, feed_vars)

    def to_pytorch(self, feed_vars=None):
        target_framework = "pytorch"
        if self.__framework == target_framework:
            return self.__feed_list
        else:
            return self._to_other(target_framework, feed_vars)

    def _to_other(self, target_framework, feed_vars=None):
        assert self.__framework == "paddle"
        assert isinstance(feed_vars, list)
        assert len(feed_vars) == len(self.__feed_list)

        feed_list = []
        for i in range(len(feed_list)):
            value = self.__feed_list[i]

            if self.__feed_spec is not None and self.__feed_spec[i].get(
                    "permute", None) is not None:
                permute_paddle2other = self.__feed_spec[i]["permute"]
                value = np.transpose(value, permute_paddle2other)

            # On dynamic mode, the check is skipped.
            if feed_vars is not None:
                # Check shape and dtype
                var = feed_list[i]
                var_shape = var.shape
                if target_framework == "tensorflow":
                    var_dtype = _convert_tensorflow_dtype(
                        var.dtype, to_string=True)
                value = check_shape_and_dtype(var_shape, var_dtype, value)

            feed_list.append(value)
        return feed_list

    @property
    def framework(self):
        return self.__framework
