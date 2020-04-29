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

name_l = []
params_l = []
result_json = []
op_dict = {}
op_fre_file = 'op_frequency.txt'


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
        '--output_direction',
        type=str,
        default=None,
        help='The direction of output json file')
    args = parser.parse_args()
    return args


def write_dict(file, op_dict, model=None):
    with open(file, 'a') as fo:
        if model is not None:
            fo.writelines(model + ' frequency: \n')
        op_list = sorted(op_dict.items(), key=lambda d: d[1], reverse=True)
        for op in op_list:
            fo.writelines(str(op[0]) + ' : ' + str(op[1]) + '\n')
        fo.writelines('\n')


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
                    op_file_dict[op] = op_file_dict.get(op, 0) + 1
                    op_dict[op] = op_dict.get(op, 0) + 1
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

            write_dict(op_fre_file, op_file_dict, file)
    write_dict(op_fre_file, op_dict, 'Summary')


def fwrite_json(args):
    dir = os.path.join(os.getcwd(), args.output_direction)
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
    file = os.listdir(dir)
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


if __name__ == '__main__':
    args = parse()
    dup(args)
    fwrite_json(args)
