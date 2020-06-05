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

import argparse
import os
import json
import copy

import clear_params


def trans_to_params_list(configs):
    op = None
    params_list = []
    for item in configs:
        if not op:
            op = item["op"]
        else:
            assert op == item["op"]
        params = item["param_info"]
        if params not in params_list:
            params_list.append(params)
    return op, params_list


def unify_use_cudnn(configs):
    op, params_list = trans_to_params_list(configs)

    unified_params_list = []
    has_use_cudnn_argument = True
    for params in params_list:
        if params not in unified_params_list:
            unified_params_list.append(params)

        # Change the use_cudnn's value to the opposite value.
        if params.get("use_cudnn", None):
            unified_params = copy.deepcopy(params)
            cudnn_param = unified_params["use_cudnn"]
            assert cudnn_param[
                "type"] == "bool", "The type of use_cudnn must be bool, but get %s." % cudnn_param[
                    "type"]
            if cudnn_param["value"] == "True":
                cudnn_param["value"] = "False"
            else:
                cudnn_param["value"] = "True"
            if unified_params not in unified_params_list:
                unified_params_list.append(unified_params)
        else:
            print("-- Operator %s does not have use_cudnn argument. Skipped." %
                  op)
            has_use_cudnn_argument = False
            break

    if has_use_cudnn_argument:
        unified_configs = []
        for params in unified_params_list:
            config = {"op": op, "param_info": params}
            unified_configs.append(config)
        return unified_configs
    else:
        return configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--json_path',
        type=str,
        default="results",
        help='Specify the path of input json. It can be a file path or a directory'
    )
    parser.add_argument(
        '--param_name',
        type=str,
        default="use_cudnn",
        help='Specify the parameter name.')
    parser.add_argument(
        '--output_dir',
        type=str,
        default="results_unified",
        help='Specify the output directory.')
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filenames = clear_params.get_json_filenames(args.json_path)
    for filename in sorted(filenames):
        try:
            configs = []
            with open(filename, 'r') as f:
                configs = json.load(f)
                print("-- Processing %s: including %d configs." %
                      (filename, len(configs)))
            unified_configs = unify_use_cudnn(configs)

            unified_filename = os.path.join(output_dir,
                                            os.path.basename(filename))
            print("-- Writing %d unified configs back to %s.\n" %
                  (len(unified_configs), unified_filename))
            with open(unified_filename, 'w') as f:
                f.write(json.dumps(unified_configs, sort_keys=True, indent=4))
        except ValueError:
            print("Cannot decode as JSON object in %s." % filename)
            break
