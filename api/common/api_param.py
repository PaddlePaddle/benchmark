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

import os
import json
import copy
import numpy as np


def parse_list(value_str, sub_dtype="int"):
    if isinstance(value_str, unicode):
        value_str = value_str.encode("utf-8")

    if sub_dtype in ["int", "int64"]:
        try:
            if value_str != "[]":
                value_str = value_str.replace("L", "").replace(
                    "[", "").replace("]", "").split(',')
                return map(int, value_str)
            else:
                return []
        except Exception as e:
            assert False, "Parse {} failed: {}".format(value_str, e)
    else:
        # TODO: check and support list of other data type.
        raise ValueError("Do not support parsing list of non-int data type.")


def parse_tuple(value_str, sub_dtype="int"):
    if sub_dtype in ["int", "int64"]:
        value_str = value_str.replace("L", "").replace("(", "").replace(
            ")", "").split(',')
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
        if isinstance(item, unicode):
            return item.encode("utf-8")
        elif isinstance(item, list):
            item_str = []
            for ele in item:
                if isinstance(ele, unicode):
                    item_str.append(ele.encode("utf-8"))
                else:
                    item_str.append(ele)
            return item_str
        else:
            return item

    def to_string(self):
        return self.name + '--' + self.type + '|' + str(self.value)

    def _translate_value(self, value_str):
        if self.type in ["float", "float32", "float64"]:
            return float(value_str)
        elif self.type in ["int", "int32", "int64"]:
            return int(value_str)
        elif self.type == "bool":
            return eval(value_str)
        elif self.type == "string":
            return None if value_str == "None" else value_str
        elif self.type == "list":
            return parse_list(value_str, sub_dtype="int")
        elif self.type == "tuple":
            return parse_tuple(value_str, sub_dtype="int")
        else:
            raise ValueError("Unsupported type \"%s\" for %s, value is %s." %
                             (self.type, self.name, value_str))


class VarParamInfo(BaseParamInfo):
    def __init__(self, name, type, dtype, shape, lod_level=0):
        self.name = self._encode_item(name)
        self.type = self._encode_item(type)
        self.dtype = self._encode_item(dtype)
        shape_str = self._encode_item(shape)
        if isinstance(shape_str, str):
            self.shape = parse_list(shape_str)
        else:
            assert isinstance(shape_str, list)
            self.shape = shape_str
        self.lod_level = self._encode_item(lod_level)

    def to_string(self):
        if self.type == "Variable":
            return self.name + "--Variable|dtype:" + str(
                self.dtype) + "|shape:" + str(self.shape)
        elif self.type == "list<Variable>":
            return self.name + "--list<Variable>"


class APIConfig(object):
    def __init__(self, op_type, params=None):
        self.__name = op_type
        self.__framework = "paddle"
        self.api_name = self.name
        self.params = params
        self.variable_list = None
        self.params_list = None
        self.backward = False
        self.feed_spec = None
        self.atol = 1e-6
        self.run_tf = True

    def alias_filename(self, filename):
        """
        Get the filename of alias config.
        If self.name = a, self.alias_config.name = b, the filename should be "dir/a.json",
        the filename of alias config will be "dir/b.json".
        """
        if hasattr(self, "alias_config"):
            dirname = os.path.dirname(filename)
            basename = os.path.basename(filename)
            basename = basename.replace(self.name, self.alias_config.name)
            return os.path.join(dirname, basename)
        return filename

    @property
    def name(self):
        return self.__name

    @property
    def framework(self):
        return self.__framework

    @property
    def alias_name(self):
        if hasattr(self, "alias_config"):
            return self.alias_config.name
        else:
            return self.name

    @property
    def alias(self):
        if hasattr(self, "alias_config"):
            return self.alias_config
        else:
            return self

    def init_from_json(self, filename, config_id=0):
        if hasattr(self, "alias_config"):
            self.alias_config.init_from_json(
                self.alias_filename(filename), config_id)
            return self

        print("---- Initialize APIConfig from %s, config_id = %d.\n" %
              (filename, config_id))
        with open(filename, 'r') as f:
            data = json.load(f)
            op = data[config_id]["op"]
            assert op == self.name or op == self.alias_name, "The op type (%s) in json file is different from the name (%s) and the alias name (%s). " \
                "The filename: %s, config_id: %d." % (
                    op, self.name, self.alias_name, filename, config_id)
            self.params = data[config_id]["param_info"]
            if data[config_id].get("atol", None) is not None:
                if isinstance(data[config_id]["atol"], str):
                    self.atol = float(data[config_id]["atol"])
                elif isinstance(data[config_id]["atol"], float):
                    self.atol = data[config_id]["atol"]

        self._parse_params()
        for param in self.params_list:
            setattr(self, param.name, param.value)
        for var in self.variable_list:
            for i in range(len(var.shape)):
                if var.shape[i] == -1:
                    var.shape[i] = 16
                var_temp = var.shape[i]
                if type(var_temp) == list:
                    for j in range(len(var_temp)):
                        if var_temp[j] == -1:
                            var_temp[j] = 16
            setattr(self, var.name + '_shape', var.shape)
            setattr(self, var.name + '_dtype', var.dtype)
        return self

    def to_tensorflow(self):
        assert self.__framework == "paddle"
        tf_config = copy.deepcopy(self)
        tf_config.__framework = "tensorflow"
        if hasattr(self, "api_list"):
            tf_config.api_name = self.api_list[self.api_name]
        if hasattr(tf_config, "alias_config"):
            tf_config.alias_config = tf_config.alias_config.to_tensorflow()
        return tf_config

    def to_string(self):
        if self.params_list is None and self.variable_list is None:
            self._parse_params()
        params_str = ""
        for var in self.variable_list:
            params_str = params_str + var.to_string() + "\n"
        for attr in self.params_list:
            params_str = params_str + attr.to_string() + "\n"
        return params_str

    def clear(self):
        for name in vars(self).keys():
            if name not in ['name', 'backward', 'feed_spec']:
                setattr(self, name, None)

    def __str__(self):
        debug_str = ('[%s][%s] %s {\n') % (self.framework, self.name,
                                           self.api_name)
        for name, value in vars(self).items():
            if name not in [
                    '_APIConfig__name', '_APIConfig__framework', 'params',
                    'api_name', 'api_list', 'variable_list', 'params_list',
                    'backward', 'feed_spec'
            ]:
                if isinstance(value, np.ndarray):
                    debug_str = debug_str + (
                        '  %s: np.ndarray(shape=%s, dtype=%s)\n') % (
                            name, str(value.shape), value.dtype)
                else:
                    debug_str = debug_str + ('  %s: %s\n') % (name, value)
        debug_str = debug_str + '}'
        return debug_str

    def _parse_params(self):
        self.variable_list = []
        self.params_list = []
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
