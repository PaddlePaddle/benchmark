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
import argparse
import warnings

import os, sys
package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_path)

from clear_params import *
from common import special_op_list

op_list = []
params_list = []
result_json = []
op_dicts_sum = {}
op_frequncy_each = {}


def get_index(lst=None, item=''):
    return [i for i in range(len(lst)) if lst[i] == item]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--json_path',
        type=str,
        default=None,
        help='The json file name or directory')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='The output directory of json files')
    parser.add_argument(
        '--op_frequency_path',
        type=str,
        default=None,
        help='The direction of output json file')
    args = parser.parse_args()
    return args


def remove_duplicates(args):
    json_path = args.json_path
    filenames = []
    if os.path.isdir(json_path):
        dir = os.path.join(os.getcwd(), json_path)
        file = os.listdir(dir)
        for f in file:
            filenames.append(os.path.join(dir, f))
    else:
        filenames = [json_path]

    for file in filenames:
        print('Read API info from json file: ' + file)
        if not os.path.isdir(file):
            op_file_dict = {}
            with open(file, 'r') as f:
                data = json.load(f)
                for i in range(0, len(data)):
                    op = data[i]["op"]
                    if op != "depthwise_conv2d" and op in special_op_list.ALIAS_OP_MAP.keys(
                    ):
                        op = special_op_list.ALIAS_OP_MAP[op]
                    param = data[i]["param_info"]
                    if check_frequency(op, param):
                        op_file_dict[op] = op_file_dict.get(op, 0) + 1
                        op_dicts_sum[op] = op_dicts_sum.get(op, 0) + 1
                    if not check_removable(op, param):
                        check_and_clear_params(op, param)
                        if op not in op_list:
                            op_list.append(op)
                            params_list.append(param)
                            result_json.append(data[i])
                        else:
                            dup_index = get_index(op_list, op)
                            dup_is = 0
                            for d_i in dup_index:
                                if params_list[d_i] == param:
                                    dup_is = 1
                            if dup_is == 0:
                                op_list.append(op)
                                params_list.append(param)
                                result_json.append(data[i])
            f.close()
            op_frequncy_each[file] = op_file_dict


def write_op_configs(args):
    # Clear the output directory.
    output_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        files = os.listdir(output_dir)
        if len(files) > 0:
            warnings.warn(
                "All %d files in output directory (%s) will be cleared and rewriten."
                % (len(files), output_dir))
        for filename in files:
            absolute_path = os.path.join(output_dir, filename)
            if os.path.getsize(absolute_path):
                os.remove(absolute_path)

    all_op_configs = {}
    for i in range(len(op_list)):
        op_type = op_list[i]
        if all_op_configs.get(op_type, None) is None:
            all_op_configs[op_type] = []
        all_op_configs[op_type].append(result_json[i])

    print("Update configs for %d operators." % len(all_op_configs))
    for op_type, config_list in sorted(all_op_configs.items()):
        op_json_path = os.path.join(output_dir, op_type + ".json")
        print("-- Write %4d configs to %s" % (len(config_list), op_json_path))
        with open(op_json_path, 'w') as f:
            f.writelines(json.dumps(config_list, sort_keys=True, indent=4))
    print("")


def write_op_frequency(args):
    def _translate_to_string(title, frequency_dict):
        frequency_str = ""
        frequency_str += "======================================================\n"
        frequency_str += "  " + title + "\n"
        frequency_str += "======================================================\n"
        frequency_dict_sorted = sorted(
            frequency_dict.items(), key=lambda d: d[1], reverse=True)
        for i in range(len(frequency_dict_sorted)):
            item = frequency_dict_sorted[i]
            frequency_str += "%s %s    %d\n" % (str(i + 1).ljust(4),
                                                item[0].ljust(36), item[1])
        frequency_str += "\n"
        return frequency_str

    if args.op_frequency_path is not None:
        op_frequency_path = args.op_frequency_path
    else:
        output_dir = os.path.abspath(args.output_dir)
        op_frequency_path = os.path.join(output_dir, 'op_frequency.txt')

    with open(op_frequency_path, 'w') as f:
        for filename, op_frequency_dict in op_frequncy_each.items():
            f.writelines(_translate_to_string(filename, op_frequency_dict))

        frequency_summary_title = "Summary of %d models" % len(
            op_frequncy_each)
        frequency_str_summary = _translate_to_string(frequency_summary_title,
                                                     op_dicts_sum)
        print(frequency_str_summary)
        f.writelines(frequency_str_summary)

    print("-- Write frequency results of %d models to %s." %
          (len(op_frequncy_each), op_frequency_path))


if __name__ == '__main__':
    args = parse_args()
    remove_duplicates(args)
    write_op_configs(args)
    write_op_frequency(args)
