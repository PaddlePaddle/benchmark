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

import json
import inspect
import importlib
import paddle.fluid as fluid

import sys
sys.path.append("..")
from common.no_need_args import *


def import_module(api_name):
    try:
        if api_name in ["embedding", "ont_hot"]:
            module_name = "paddle.fluid"
        else:
            module_name = "paddle.fluid.layers"
        module = importlib.import_module(module_name)
        return getattr(module, api_name)
    except ImportError:
        print("Cannot immport %s.%s." % (module_name, api_name))
        return None


def check_and_clear_params(api_name, params, print_detail=False):
    func = import_module(api_name)
    if func is not None:
        argspec = inspect.getargspec(func)
        if print_detail:
            print("API:", api_name, ",", argspec)

        no_need_args = []
        if api_name in NO_NEED_ARGS.keys():
            no_need_args = NO_NEED_ARGS[api_name]
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
                    print("   Remove %s (type: %s, value: %s)." %
                          (name, content["type"], content["value"]))
                params.pop(name)

        no_need_args.remove("name")


if __name__ == '__main__':
    op = "conv2d"
    filename = "results/" + op + ".json"
    cleared_filename = "results_cleared/" + op + ".json"
    with open(filename, 'r') as f:
        data = json.load(f)
        print("-- Processing %s: including %d configs." %
              (filename, len(data)))
        for i in range(0, len(data)):
            check_and_clear_params(
                data[i]["op"], data[i]["param_info"], print_detail=True)
    print("-- Writing %d cleared configs back to %s." %
          (len(data), cleared_filename))
    with open(cleared_filename, 'w') as f:
        f.write(json.dumps(data, sort_keys=True, indent=4))
