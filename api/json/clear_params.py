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

from __future__ import print_function

import os
import json
import inspect
import importlib
import paddle.fluid as fluid

import sys
sys.path.append("..")
from common import special_op_list


def import_layers(api_name):
    def _import_func(module_name, api_name):
        try:
            module = importlib.import_module(module_name)
            func = getattr(module, api_name)
            # print("Successly import %s.%s" % (module_name, api_name))
            return func
        except Exception:
            # print("Failed to import %s.%s" % (module_name, api_name))
            return None

    if api_name in special_op_list.ALIAS_OP_MAP.keys():
        api_name = special_op_list.ALIAS_OP_MAP[api_name]

    all_module_names = [
        "paddle.fluid", "paddle.fluid.layers", "paddle.fluid.contrib.layers",
        "paddle"
    ]
    for module_name in all_module_names:
        func = _import_func(module_name, api_name)
        if func is not None:
            break
    return func


def check_removable(api_name, params):
    if not isinstance(params, dict) or not params:
        return True
    if api_name in special_op_list.CONTROL_FLOW_OPS:
        return True
    if api_name in special_op_list.EXCLUDE_OPS:
        return True
    return False


def check_frequency(api_name, params):
    if api_name in special_op_list.CONTROL_FLOW_OPS:
        return True
    if not isinstance(params, dict) or not params:
        return False
    if api_name in special_op_list.EXCLUDE_OPS:
        return False
    return True


def check_and_clear_params(api_name, params, print_detail=False):
    func = import_layers(api_name)
    assert func is not None, "Cannot import %s from paddle.fluid.layers and paddle" % api_name

    if func is not None:
        argspec = inspect.getargspec(func)
        if print_detail:
            print("   OP:", api_name, ",", argspec)

        no_need_args = []
        if api_name in special_op_list.NO_NEED_ARGS.keys():
            no_need_args = special_op_list.NO_NEED_ARGS[api_name]
            if print_detail:
                print(no_need_args)
                print(type(no_need_args))
        no_need_args.append("name")

        for arg_name in argspec.args:
            if arg_name not in no_need_args and arg_name not in params.keys():
                if print_detail:
                    print("   Argument %s is not set." % (arg_name))

        for name, content in params.items():
            if name not in argspec.args or name in no_need_args:
                if print_detail:
                    if content["type"] == "Variable":
                        print("   Remove %s (type: %s, dtype: %s, shape: %s)."
                              % (name, content["type"], content["dtype"],
                                 content["shape"]))
                    else:
                        print("   Remove %s (type: %s, value: %s)." %
                              (name, content["type"], content["value"]))
                params.pop(name)

        no_need_args.remove("name")


def get_json_filenames(config_path):
    abs_path = os.path.abspath(config_path)
    if os.path.isdir(abs_path):
        filenames = []
        files = os.listdir(abs_path)
        print("There are %d configs under %s." % (len(files), abs_path))
        for f in files:
            filenames.append(os.path.join(abs_path, f))
    else:
        filenames = [config_path]
    return filenames


if __name__ == '__main__':
    config_path = "results_all"
    output_dir = os.path.abspath("results_all_cleared")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filenames = get_json_filenames(config_path)
    for filename in filenames:
        try:
            data = []
            with open(filename, 'r') as f:
                data = json.load(f)
                print("-- Processing %s: including %d configs." %
                      (filename, len(data)))
                remove_list = []
                for i in range(0, len(data)):
                    op = data[i]["op"]
                    params = data[i]["param_info"]
                    if not check_removable(op, params):
                        check_and_clear_params(op, params, print_detail=True)
                    else:
                        remove_list.append(data[i])
                for item in remove_list:
                    data.remove(item)

            if data:
                cleared_filename = os.path.join(output_dir,
                                                os.path.basename(filename))
                print("-- Writing %d cleared configs back to %s.\n" %
                      (len(data), cleared_filename))
                with open(cleared_filename, 'w') as f:
                    f.write(json.dumps(data, sort_keys=True, indent=4))
        except ValueError:
            print("Cannot decode as JSON object in %s." % filename)
            break
