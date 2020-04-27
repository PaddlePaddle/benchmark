import json
import os
import argparse

name_l = []
params_l = []
name_dep_json=[]
param_dep_json=[]
op_dict = {}
op_fre_file = 'op_frequency.txt'

def get_index(lst=None, item=''):
    return [i for i in range(len(lst)) if lst[i] == item]

def parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help='The models name')
    args = parser.parse_args()
    filename_list = args.model_name
    return filename_list

def write_dict(file, op_dict, model=None):
    with open(file, 'a') as fo:
        if model is not None:
            fo.writelines(model +' frequency: \n')
        op_list = sorted(op_dict.items(), key = lambda d:d[1])
        for op in op_list:
            fo.writelines(str(op[0])+' : '+ str(op[1]) +'\n')
        fo.writelines('\n')

def dup(filename_list):
    fileneme = filename_list.split(',')
    for file in fileneme:
        filen = file + '_api_info.json'

        op_file_dict = {}
        with open(filen, 'r') as f:
            data = json.load(f)
            for i in range(0, len(data)):
                op = data[i]["op"]
                pa = data[i]["param_info"]
                op_file_dict[op] = op_file_dict.get(op, 0) + 1
                op_dict[op] = op_dict.get(op, 0) + 1
                if op not in name_l:
                    name_l.append(op)
                    params_l.append(pa)
                    name_dep_json.append(json.dumps(op))
                    param_dep_json.append(json.dumps(pa))
                else:
                    dup_index = get_index(name_l, op)
                    dup_is = 0
                    for d_i in dup_index:
                        if params_l[d_i] == pa:
                            dup_is = 1
                    if dup_is == 0:
                        name_l.append(op)
                        params_l.append(pa)
                        name_dep_json.append(json.dumps(op))
                        param_dep_json.append(json.dumps(pa))

        write_dict(op_fre_file, op_file_dict, file)
    write_dict(op_fre_file, op_dict, 'Summary')

def fwrite_json():
    for i in range(len(name_dep_json)):
        op = name_l[i]
        with open('result/' + op + '.json', 'a') as fw:
            fw.writelines('{\n"op": ' + name_dep_json[i] + ',\n')
            fw.writelines('"param_info":' + param_dep_json[i]+'},\n')

if __name__ == '__main__':
    filen = parse()
    dup(filen)
    fwrite_json()
