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
        self.input_list = []
        self.params_list = []
        self.backward = False

    def __str__(self):
        debug_str = ('API params of <%s> {\n') % (self.name)
        for name, value in vars(self).items():
            if name not in [
                    'name', 'params', 'input_list', 'params_list', 'backward'
            ]:
                debug_str = debug_str + ('  %s: %s\n') % (name, value)
        debug_str = debug_str + '}'
        return debug_str

    def init_from_json(self, filename, pos=0):
        with open(filename, 'r') as f:
            data = json.load(f)
            self.name = data[pos]["op"]
            self.params = data[pos]["param_info"]
            self.convert_params_list()
            for params in self.params_list:
                self.dy_param(
                    params.name.encode('utf-8'),
                    params.type.encode('utf-8'), params.value.encode('utf-8'))
            for input_p in self.input_list:
                shape = input_p.shape.encode('utf-8')
                shape = shape.replace("L", "").replace("[", "").replace(
                    "]", "").split(',')
                self.dy_input_param(pos,
                                    input_p.name.encode('utf-8'),
                                    input_p.dtype.encode('utf-8'), shape,
                                    input_p.lod_level)
        return self

    def to_tensorflow(self):
        pass

    def _convert_params_to_str(self):
        params = ""
        for p_key, p_value in self.params.items():
            params_str = ""
            param_name = p_key
            dtype = p_value["dtype"]
            type = ""
            v_type = 0
            for v_k in p_value.keys():
                if v_k == "type":
                    v_type = 1
            if v_type == 1:
                type = p_value["type"]
                shape = p_value["shape"]
                var_ = VarParamInfo(param_name, type, dtype, shape)
            else:
                value = p_value["value"]
                var_ = BaseParamInfo(param_name, dtype, value)
            params_str = var_.convert_params_to_str()
            params = params + params_str
        return params

    def convert_params_list(self):
        for p_key, p_value in self.params.items():
            param_name = p_key
            dtype = p_value["dtype"]
            type = ""
            v_type = 0
            for v_k in p_value.keys():
                if v_k == "type":
                    v_type = 1
            if v_type == 1:
                type = p_value["type"]
                shape = p_value["shape"]
                var_ = VarParamInfo(param_name, type, dtype, shape)
                self.input_list.append(var_)
            else:
                value = p_value["value"]
                var_ = BaseParamInfo(param_name, dtype, value)
                self.params_list.append(var_)
        return self

    def dy_param(self, name, type, value):
        if type == "float":
            value_t = float(value)
            setattr(self, name, value_t)
        elif type == "int":
            value_t = int(value)
            setattr(self, name, value_t)
        elif type == "bool":
            value_t = bool(value)
            setattr(self, name, value_t)
        elif type == "string":
            if value == "None":
                value_t = None
            else:
                value_t = value
            setattr(self, name, value_t)
        return self

    def dy_input_param(self, pos, name, dtype, shape, lod_level):
        setattr(self, name + '_shape', map(int, shape))
        setattr(self, name + '_dtype', dtype)
        return self
