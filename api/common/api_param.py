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
    def __init__(self, op_type, params):
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
            self._dy_param(
                params.name.encode('utf-8'),
                params.type.encode('utf-8'), params.value.encode('utf-8'))
        for input_p in self.variable_list:
            self._dy_input_param(
                input_p.name.encode('utf-8'),
                input_p.dtype.encode('utf-8'),
                input_p.shape.encode('utf-8'), input_p.lod_level)
        return self

    def to_tensorflow(self):
        return self

    def _convert_params_to_str(self):
        params = ""
        for p_key, p_value in self.params.items():
            param_name = p_key
            dtype = p_value["dtype"]
            if self._is_variable(p_value):
                type = p_value["type"]
                shape = p_value["shape"]
                var_ = VarParamInfo(param_name, type, dtype, shape)
            else:
                value = p_value["value"]
                var_ = BaseParamInfo(param_name, dtype, value)
            params = params + var_.convert_params_to_str()
        return params

    def _parse_params(self):
        for p_key, p_value in self.params.items():
            param_name = p_key
            dtype = p_value["dtype"]
            if self._is_variable(p_value):
                type = p_value["type"]
                shape = p_value["shape"]
                var_ = VarParamInfo(param_name, type, dtype, shape)
                self.variable_list.append(var_)
            else:
                value = p_value["value"]
                var_ = BaseParamInfo(param_name, dtype, value)
                self.params_list.append(var_)

    def _parse_list(self, shape_str):
        shape_str = shape_str.replace("L", "").replace("[", "").replace(
            "]", "").split(',')
        # TODO: check and support list of other data type.
        return map(int, shape_str)

    def _is_variable(self, value):
        if value.get("type", None) is not None and value["type"] == "Variable":
            return True
        return False

    def _dy_param(self, name, type, value):
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
            value_t = self._parse_list(value)
        setattr(self, name, value_t)

    def _dy_input_param(self, name, dtype, shape, lod_level):
        setattr(self, name + '_shape', self._parse_list(shape))
        setattr(self, name + '_dtype', dtype)
