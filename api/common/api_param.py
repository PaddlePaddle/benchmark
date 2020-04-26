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

import json


class BaseParamInfo(object):
    def __init__(self, name, type, value):
        self.name = name
        self.type = type
        self.value = value

    def convert_params_to_str(self):
        param = self.name + '--' + self.type + '| ' + self.value + '\n '
        return param


class VarParamInfo(object):
    def __init__(self, name, type, dtype, shape, lod_level=0):
        self.name = name
        self.type = type
        self.dtype = dtype
        self.shape = shape
        self.lod_level = lod_level

    def convert_params_to_str(self):
        param = self.name + '--' + self.type + '| ' + self.dtype + '| shape:' + self.shape + '\n '
        return param


class APIConfig(object):
    def __init__(self, op_type, params=None):
        self.name = op_type
        self.params = params
        self.variable_list = []
        self.params_list = []
        self.backward = False
        self.feed_spec = None

    def __str__(self):
        debug_str = ('API params of <%s> {\n') % (self.name)
        for name, value in vars(self).items():
            if name not in [
                    'name', 'params', 'variable_list', 'params_list',
                    'backward', 'feed_spec'
            ]:
                debug_str = debug_str + ('  %s: %s\n') % (name, value)
        debug_str = debug_str + '}'
        return debug_str

    def init_from_json(self, filename, config_id=0):
        with open(filename, 'r') as f:
            data = json.load(f)
            self.name = data[config_id]["op"]
            self.params = data[config_id]["param_info"]

        self._parse_params()
        for params in self.params_list:
            self._translate_param(
                name=params.name.encode('utf-8'),
                type=params.type.encode('utf-8'),
                value=params.value.encode('utf-8'))
        for input_p in self.variable_list:
            self._translate_variable(
                name=input_p.name.encode('utf-8'),
                dtype=input_p.dtype.encode('utf-8'),
                shape=input_p.shape.encode('utf-8'),
                lod_level=input_p.lod_level)
        return self

    def to_tensorflow(self):
        return self

    def _convert_params_to_str(self):
        params = ""
        for name, value in self.params.items():
            if self._is_variable(value):
                info = VarParamInfo(name, value["type"], value["dtype"],
                                    value["shape"])
            else:
                info = BaseParamInfo(name, value["type"], value["value"])
            params = params + info.convert_params_to_str()
        return params

    def _parse_params(self):
        for name, value in self.params.items():
            if self._is_variable(value):
                info = VarParamInfo(name, value["type"], value["dtype"],
                                    value["shape"])
                self.variable_list.append(info)
            else:
                info = BaseParamInfo(name, value["type"], value["value"])
                self.params_list.append(info)

    def _parse_list(self, value_str, dtype):
        value_str = value_str.replace("L", "").replace("[", "").replace(
            "]", "").split(',')
        # TODO: check and support list of other data type.
        if dtype == "int":
            return map(int, value_str)
        else:
            raise ValueError(
                "Do not support parsing list of non-int data type.")

    def _is_variable(self, value):
        if value.get("type", None) is not None and value["type"] == "Variable":
            return True
        return False

    def _translate_param(self, name, type, value):
        value_t = None
        if type in ["float", "float32", "float64"]:
            value_t = float(value)
        elif type in ["int", "int32", "int64"]:
            value_t = int(value)
        elif type == "bool":
            value_t = bool(value)
        elif type == "string":
            if value == "None":
                value_t = None
            else:
                value_t = value
        elif type == "list":
            value_t = self._parse_list(value, dtype="int")
        setattr(self, name, value_t)

    def _translate_variable(self, name, dtype, shape, lod_level):
        setattr(self, name + '_shape', self._parse_list(shape, dtype="int"))
        setattr(self, name + '_dtype', dtype)
