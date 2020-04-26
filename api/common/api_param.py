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


def parse_list(value_str, sub_dtype="int"):
    if sub_dtype in ["int", "int64"]:
        value_str = value_str.replace("L", "").replace("[", "").replace(
            "]", "").split(',')
        return map(int, value_str)
    else:
        # TODO: check and support list of other data type.
        raise ValueError("Do not support parsing list of non-int data type.")


class BaseParamInfo(object):
    def __init__(self, name, type, value):
        self.name = self._encode_item(name)
        self.type = self._encode_item(type)
        self.value = self._translate_value(self._encode_item(value))

    def _encode_item(self, item):
        return item.encode("utf-8") if isinstance(item, str) else item

    def to_string(self):
        return self.name + '--' + self.type + '| ' + str(self.value) + '\n '

    def _translate_value(self, value):
        if self.type in ["float", "float32", "float64"]:
            return float(value)
        elif self.type in ["int", "int32", "int64"]:
            return int(value)
        elif self.type == "bool":
            return bool(value)
        elif self.type == "string":
            return None if value == "None" else value
        elif self.type == "list":
            return parse_list(value, sub_dtype="int")


class VarParamInfo(BaseParamInfo):
    def __init__(self, name, type, dtype, shape, lod_level=0):
        self.name = self._encode_item(name)
        self.type = self._encode_item(type)
        self.dtype = self._encode_item(dtype)
        if isinstance(shape, str):
            self.shape = parse_list(self._encode_item(shape))
        else:
            assert isinstance(shape, list)
            self.shape = shape
        self.lod_level = self._encode_item(lod_level)

    def to_string(self):
        return self.name + '--' + self.type + '| ' + str(
            self.dtype) + '| shape:' + str(self.shape) + '\n '


class APIConfig(object):
    def __init__(self, op_type, params=None):
        self.name = op_type
        self.params = params
        self.variable_list = []
        self.params_list = []
        self.backward = False
        self.feed_spec = None

    def init_from_json(self, filename, config_id=0):
        with open(filename, 'r') as f:
            data = json.load(f)
            self.name = data[config_id]["op"]
            self.params = data[config_id]["param_info"]

        self._parse_params()
        for param in self.params_list:
            setattr(self, param.name, param.value)
        for var in self.variable_list:
            setattr(self, var.name + '_shape', var.shape)
            setattr(self, var.name + '_dtype', var.dtype)
        return self

    def to_tensorflow(self):
        return self

    def to_string(self):
        self._parse_params()
        params = ""
        for info in self.variable_list:
            params = params + info.to_string()
        for info in self.params_list:
            params = params + info.to_string()
        return params

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

    def _parse_params(self):
        for name, value in self.params.items():
            assert value.get("type", None) is not None
            if value["type"] == "Variable":
                info = VarParamInfo(name, value["type"], value["dtype"],
                                    value["shape"])
                self.variable_list.append(info)
            elif value["type"] == "list<Variable>":
                dtype_list = []
                shape_list = []
                for key, var in value.items():
                    if key != "type":
                        assert var["type"] == "Variable"
                        dtype_list.append(var["dtype"])
                        shape_list.append(parse_list(var["shape"]))
                info = VarParamInfo(name, value["type"], dtype_list,
                                    shape_list)
                self.variable_list.append(info)
            else:
                info = BaseParamInfo(name, value["type"], value["value"])
                self.params_list.append(info)
