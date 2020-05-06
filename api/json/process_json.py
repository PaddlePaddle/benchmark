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
import os
import argparse

import sys
sys.path.append("..")
from clear_params import *
from common import special_op_list

op_list = []
params_list = []
result_json = []
op_dicts_sum = {}
op_frequncy_each = {}
op_frequncy_file = 'op_frequency.txt'


def get_index(lst=None, item=''):
    return [i for i in range(len(lst)) if lst[i] == item]


def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--json_path',
        type=str,
        default=None,
        help='The json file name or direction')
    parser.add_argument(
        '--output_dir',
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


def fwrite_json(args):
    dir = os.path.join(os.getcwd(), args.output_dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        file = os.listdir(dir)
        for fi in file:
            if os.path.getsize(os.path.join(dir, fi)):
                os.remove(os.path.join(dir, fi))
    for i in range(len(op_list)):
        op = (op_list[i] + '.json').encode("utf-8")
        with open(os.path.join(dir, op), 'a') as fw:
            fw.writelines(
                json.dumps(
                    result_json[i], sort_keys=True, indent=4) + ',\n')
        fw.close()
    file = os.listdir(dir)
    print('The json files will be rewrite: ')
    for i in file:
        with open(os.path.join(dir, i), 'r+') as f:
            print('The json file after processing: ' + os.path.join(dir, i))
            content = f.read()
            f.seek(0, 0)
            f.write('[\n' + content)
            f.seek(-2, 2)
            f.truncate()
            f.seek(0, 2)
            f.truncate()
            f.write('\n]\n')
        f.close()
        with open(os.path.join(dir, i), 'r') as fs:
            data = json.load(fs)
        fs.close()
        print('The number of params: ' + str(len(data)))


def write_dict(args):
    dir = os.path.join(os.getcwd(), args.output_dir)
    with open(os.path.join(dir, op_frequncy_file), 'w') as fre:
        for file, op_dicts in op_frequncy_each.items():
            if file is not None:
                fre.writelines(file + ' frequency: \n')
            op_dicts_sorted = sorted(
                op_dicts.items(), key=lambda d: d[1], reverse=True)
            for op in op_dicts_sorted:
                fre.writelines(str(op[0]) + ' : ' + str(op[1]) + '\n')
            fre.writelines('\n\n')
        fre.writelines('Summary frequency: \n')
        op_dicts_sum_sorted = sorted(
            op_dicts_sum.items(), key=lambda d: d[1], reverse=True)
        for op in op_dicts_sum_sorted:
            fre.writelines(str(op[0]) + ' : ' + str(op[1]) + '\n')
        fre.writelines('\n')
    fre.close()
    print('The op frequency file: ' + os.path.join(dir, op_frequncy_file))


if __name__ == '__main__':
    args = parse()
    remove_duplicates(args)
    fwrite_json(args)
    write_dict(args)
