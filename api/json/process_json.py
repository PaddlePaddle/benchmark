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

name_l = []
params_l = []
result_json = []
op_dict = {}
op_file_whole = {}
op_fre_file = 'op_frequency.txt'
control_op = [
    'while', 'switch', 'less_than', 'less_equal', 'greater_than',
    'greater_equal', 'equal', 'not_equal', 'cond', 'ifelse', 'dynamic_rnn',
    'static_rnn', 'case', 'switch_case', 'while_loop'
]


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


def dup(args):
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
                    pa = data[i]["param_info"]
                    if pa != '' and pa != {} or op in control_op:
                        op_file_dict[op] = op_file_dict.get(op, 0) + 1
                        op_dict[op] = op_dict.get(op, 0) + 1
                    if pa != '' and pa != {}:
                        check_and_clear_params(op, pa)
                        if op not in name_l:
                            name_l.append(op)
                            params_l.append(pa)
                            result_json.append(data[i])
                        else:
                            dup_index = get_index(name_l, op)
                            dup_is = 0
                            for d_i in dup_index:
                                if params_l[d_i] == pa:
                                    dup_is = 1
                            if dup_is == 0:
                                name_l.append(op)
                                params_l.append(pa)
                                result_json.append(data[i])
            f.close()
            op_file_whole[file] = op_file_dict


def fwrite_json(args):
    dir = os.path.join(os.getcwd(), args.output_dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        file = os.listdir(dir)
        for fi in file:
            if os.path.getsize(os.path.join(dir, fi)):
                os.remove(os.path.join(dir, fi))
    for i in range(len(name_l)):
        op = (name_l[i] + '.json').encode("utf-8")
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
    with open(os.path.join(dir, op_fre_file), 'w') as fre:
        for file_n, op_dicts in op_file_whole.items():
            if file_n is not None:
                fre.writelines(file_n + ' frequency: \n')
            op_list = sorted(
                op_dicts.items(), key=lambda d: d[1], reverse=True)
            for op in op_list:
                fre.writelines(str(op[0]) + ' : ' + str(op[1]) + '\n')
        fre.writelines('\nSummary frequency: \n')
        op_list = sorted(op_dict.items(), key=lambda d: d[1], reverse=True)
        for op in op_list:
            fre.writelines(str(op[0]) + ' : ' + str(op[1]) + '\n')
        fre.writelines('\n')
    fre.close()
    print('The op frequency file: ' + os.path.join(dir, op_fre_file))


if __name__ == '__main__':
    args = parse()
    dup(args)
    fwrite_json(args)
    write_dict(args)
