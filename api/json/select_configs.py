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

import argparse
import os
import json
import warnings
from operator import mul
from functools import reduce
import random
import copy

import sys
sys.path.append("..")
from common.api_param import parse_list


def select_configs_by_json(args, origin_configs, configs_without_input,
                           all_shapes, input_type):
    """
    Select configs according to json config and save the selected
    configs to the sepcified json file.
    """
    ignored_params = []
    if args.ignored_params:
        ignored_params += args.ignored_params
    all_config_info = []
    for config in configs_without_input:
        config_info = ""
        for param in config["param_info"]:
            if param not in ignored_params:
                value = config["param_info"][param]["value"]
                config_info += param + "=" + value + " "
        all_config_info.append(config_info)
    config_groups = grouping_configs(all_config_info, all_shapes, input_type)
    all_selected_ids = select_and_print_configs(config_groups, all_shapes)

    out_dir = os.path.dirname(args.output_json_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    configs = []
    for index in all_selected_ids:
        configs.append(origin_configs[index])
    with open(args.output_json_file, 'w') as f:
        json.dump(configs, f, indent=4, sort_keys=True)


def select_configs_by_log(args, forward_logs, backward_logs):
    """
    Select configs according to forward logs and backward logs and save the selected
    configs to the sepcified json file.

    Args:
        args(object): An object to take the attributes.
        forward_logs(list): A list of forward logs.
        backward_logs(list): A list of backward logs.
    """
    ignored_params = ["op"]
    input_shape = ["input_shape"]
    if args.input_shape:
        input_shape = args.input_shape
    if args.ignored_params:
        ignored_params += args.ignored_params
    shapes_list = get_input_shapes(forward_logs, input_shape)
    removed_params = ignored_params + input_shape
    combined_logs = combine_logs_with_key_params(removed_params, forward_logs,
                                                 backward_logs)
    config_groups = grouping_configs(combined_logs, shapes_list)
    all_selected_ids = select_and_print_configs(config_groups, shapes_list)

    with open(args.input_json_file, 'r') as f:
        all_configs = json.load(f)
    out_dir = os.path.dirname(args.output_json_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    configs = []
    for index in all_selected_ids:
        configs.append(all_configs[index])
    with open(args.output_json_file, 'w') as f:
        json.dump(configs, f, indent=4, sort_keys=True)


def select_and_print_configs(config_groups, shapes_list):
    print("=" * 30 + "config_groups" + "=" * 30)
    all_selected_ids = []
    i = 0
    for key in config_groups:
        print("config {0}: {1}, total: {2}.".format(
            i, key, len(config_groups[key]['ids'])))
        shape_groups = config_groups[key]['shape_groups']
        if not shape_groups:
            selected_ids = random.sample(config_groups[key]['ids'], 1)
            all_selected_ids.extend(selected_ids)
            select_ids_info = "  Select {0} config_ids: {1}.".format(
                len(selected_ids), selected_ids)
            print(select_ids_info)
        else:
            all_selected_ids.extend(
                select_from_shape_groups(shape_groups, shapes_list))
        i += 1
    return all_selected_ids


def select_from_shape_groups(shape_groups, shapes):
    """
    Select configs from shape groups. The small, middle and large input will
    be selected.

    args:
        shape_groups(dict): A dict that groups input shapes.
        shapes(list): A list of input shapes. 

    Returns: A list of selected config ids.
    """
    all_selected_ids = []
    j = 0
    for label in shape_groups:
        candidate_ids = []
        selected_ids = []
        ids = shape_groups[label]['ids']
        ids = rearrange_ids(shape_groups[label]['sizes'], ids)
        if len(ids) <= 3:
            candidate_ids = ids
        else:
            candidate_ids = [ids[0], ids[int(len(ids) / 2)], ids[-1]]
        selected_shapes = []
        for idx in candidate_ids:
            if shapes[idx] not in selected_shapes:
                selected_shapes.append(shapes[idx])
                selected_ids.append(idx)
        all_selected_ids += selected_ids
        selected_shapes_info = " The shapes are: "
        for shape in selected_shapes:
            selected_shapes_info += "{} ".format(shape)
        shape_groups_info = " " * 2 + "shape {0}: {1}, total: {2}.".format(
            j, label, len(ids))
        select_ids_info = " Select {0} config_ids: {1}.".format(
            len(selected_ids), selected_ids)
        print(shape_groups_info + select_ids_info + selected_shapes_info)
        j += 1

    return all_selected_ids


def combine_logs_with_key_params(removed_params, forward_logs, backward_logs):
    """
    Combine each forward log with the corresponding backward log. First, some params in
    forward and backward logs are removed. Second, the union of each forward and
    corresponding backward log is computed.

    Args:
        removed_params(list): A list of removed params. It usually contains "op" and "input_shape".
        forward_logs(list): A list of forward logs.
        backward_logs(list): A list og backward logs.

    Returns: A list of combined logs.
    """
    for logs in [forward_logs, backward_logs]:
        for i in range(len(logs)):
            logs[i] = remove_params(logs[i], removed_params)

    combined_logs = forward_logs
    for i in range(len(forward_logs)):
        if forward_logs[i] != backward_logs[i]:
            intersection = list(
                set(forward_logs[i]).intersection(set(backward_logs[i])))
            difference = list(
                set(forward_logs[i]).symmetric_difference(
                    set(backward_logs[i])))
            combined_logs[i] = intersection + difference
        combined_logs[i] = ' '.join(combined_logs[i])

    return combined_logs


def grouping_configs(logs, shapes, input_type="Variable"):
    """
    Groups all configs according to the logs. First, all configs are grouped by key params
    without input shape. Second, the results of first step are grouped by input shape.

    Args:
        logs(list): A list of logs in which each item combines forward and backward log.
        shapes(list): A list of input shapes.

    Returns: A 2-D dict of config groups.
    """
    config_groups = dict()
    # group all configs by key params without input shape.
    for i in range(len(logs)):
        if logs[i] not in config_groups.keys():
            config_groups[logs[i]] = {'ids': [i]}
        else:
            config_groups[logs[i]]['ids'] += [i]

    # group config_groups by input shape.
    for key in config_groups:
        config_ids = config_groups[key]['ids']
        shape_groups = group_input_shapes(shapes, config_ids, input_type)
        config_groups[key]['shape_groups'] = shape_groups

    return config_groups


def remove_params(log, removed_params):
    """
    Remove params from logs according to the names of removed params.

    Args:
        log(list): A list of logs in which each item is a string.
        removed_params(list): The names of removed params.

    Example:
        los=['op=conv data_format=NCHW filter_size=[1, 1]']
        removed_params=['op']
        result=['data_format=NCHW filter_size=[1, 1]']

    Returns: A list of logs.
    """
    result = list(log)
    if removed_params:
        for item in log:
            param_val = item.split("=")
            if param_val[0] in removed_params:
                result.remove(item)

    return result


def get_input_shapes(logs, input_shape):
    """
    Get input shapes from logs and parse them into lists.

    Args:
        logs(list): A list of forward logs or backward logs. It is used to extract
            input shapes.
        input_shape(list): The name of input shape in logs. For example, if input_shape=[10, 10]
            or x_shape=[10, 10] in logs. Then the name of input shape is "input_shape" or "x_shape".

    Returns: A list of input shapes and each input shape is also a list.
    """

    all_shapes = []
    for log in logs:
        shapes = []
        for item in log:
            param_val = item.split("=")
            if param_val[0] in input_shape:
                shape = parse_list(param_val[1])
                shapes.append(list(shape))
        all_shapes.append(shapes)

    return all_shapes


def group_input_shapes(shapes, config_ids, input_type):
    """
    Group the input shapes according to the shape's label.

    Args: 
        shapes(list): A list of input shapes.
        config_ids(list): A list of config ids in one group.

    Returns: A 2-D dict of shape groups.
    """
    shape_groups = dict()
    if len(shapes) == 0:
        warnings.warn("Group configs regardless of input shape.")
        return shape_groups
    for index in config_ids:
        shape = shapes[index]
        label, size = label_shape(shape, input_type)
        if label not in shape_groups.keys():
            shape_groups[label] = {'ids': [index], 'sizes': [size]}
        else:
            shape_groups[label]['ids'] += [index]
            shape_groups[label]['sizes'] += [size]

    return shape_groups


def get_input_shapes_from_json(args, origin_configs):
    configs_without_input = []
    all_shapes = []
    input_type = "Variable"
    input_name = ["input", "x", "y"]
    for config in origin_configs:
        config_res = copy.deepcopy(config)
        input_shapes = []
        var_shapes = []
        for name, value in config["param_info"].items():
            if args.ignored_params is not None and name in args.ignored_params:
                continue
            if value["type"] in ["Variable", "numpy.ndarray"]:
                if value["type"] == "Variable":
                    shape = list(parse_list(value["shape"]))
                else:
                    shape = list(parse_list(value["value"]))
                for i in range(len(shape)):
                    if shape[i] == -1:
                        shape[i] = 16
                if name in input_name:
                    input_shapes.append(shape)
                else:
                    var_shapes.append(shape)
                del config_res["param_info"][name]
            elif value["type"] == "list<Variable>":
                input_type = "list<Variable>"
                for key, var in value.items():
                    if key != "type":
                        shape = list(parse_list(var["shape"]))
                        for i in range(len(shape)):
                            if shape[i] == -1:
                                shape[i] = 16
                        input_shapes.append(shape)
                del config_res["param_info"][name]
        configs_without_input.append(config_res)
        if len(input_shapes) != 0:
            all_shapes.append(input_shapes)
        elif len(var_shapes) != 0 and len(var_shapes) <= 2:
            all_shapes.append(var_shapes)

    return configs_without_input, all_shapes, input_type


def label_shape(shape, input_type):
    """
    Label shape with the features. When only one shape is found, label the shape
    according to the number of dimensions and whether the size is a power of 2. 
    When two shape is found, if the shapes are the same, label the shapes according
    to the rule of one input, if the shapes are not the same, label the shapes
    according to the number of dimensions.

    Args:
        shape: A list of input shapes. Each item is also a list containing 1 or 2 items.

    Returns: A label of shapes that is a string and the size of the input with the most
        dimensions.
    """
    if input_type == "list<Variable>":
        is_same_shape = 'T'
        for s in shape:
            if s != shape[0]:
                is_same_shape = 'F'
                break
        label = 'is_same_shape=' + is_same_shape
        all_sizes = [reduce(mul, shape[i]) for i in range(len(shape))]
        size = max(all_sizes)
    elif len(shape) == 1:
        size = reduce(mul, shape[0])
        is_power_of_2 = 'T' if size & (size - 1) == 0 else 'F'
        label = str(len(shape[0])) + '-D' + ' is_power_of_2=' + is_power_of_2
    elif len(shape) == 2:
        if (shape[0] == shape[1]):
            is_same_shape = 'T'
            size = reduce(mul, shape[0])
            is_power_of_2 = 'T' if size & (size - 1) == 0 else 'F'
            label = 'is_same_shape=' + is_same_shape + ' ' + str(
                len(shape[0])) + '-D-' + str(len(shape[
                    1])) + '-D' + ' is_power_of_2=' + is_power_of_2
        else:
            is_same_shape = 'F'
            size1 = reduce(mul, shape[0])
            size2 = reduce(mul, shape[0])
            label = 'is_same_shape=' + is_same_shape + ' ' + str(
                len(shape[0])) + '-D-' + str(len(shape[1])) + '-D'
            size = size1 if (len(shape[0]) > len(shape[1])) else size2

    return label, size


def rearrange_ids(sizes, ids):
    """
    This function will sort sizes by ascending order and use the index of sorted sizes
    to rearrange ids. 

    Example: 
        sizes = [400, 256, 1000, 512]
        ids = [0, 3, 16, 10]

        sorted_sizes = [256, 400, 512, 1000]
        sorted_ids = [1, 0, 3, 2]
        return: [3, 0, 10, 16]

    Args: 
        sizes(list): A list of sizes to be sorted.
        ids(list): A list of config ids to be rearrange by index of sorted sizes. 
    
    Returns: A list of rearranged ids.
    """
    sorted_nums = sorted(enumerate(sizes), key=lambda x: x[1])
    sorted_ids = [i[0] for i in sorted_nums]
    return [ids[idx] for idx in sorted_ids]


def get_logs(op_name, log_file):
    """
    Given the name of the OP and the path of the log, split forward and backward logs.

    Args: 
        op_name(str): The OP's name that is used to extract key logs.
        log_file(str): The path of Op's log.

    Returns: Two lists of forward and backward logs.
    """
    forward_logs = []
    backward_logs = []
    forward_name = "op=" + op_name
    backward_name = forward_name + '_grad'
    with open(log_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if backward_name in line:
                line = line.replace(', ', ',').split(' ')
                backward_logs.append(line)
            elif forward_name in line:
                line = line.replace(', ', ',').split(' ')
                forward_logs.append(line)
    if not forward_logs:
        raise ValueError("Could not find {0} in {1}.".format(forward_name,
                                                             log_file))
    if not backward_logs:
        warnings.warn("Only forward logs are used to select configs.")
    else:
        if len(forward_logs) != len(backward_logs):
            raise ValueError(
                "There are {0} logs containing {1}, but {2} logs containing {3}. They should be equal".
                format(
                    len(forward_logs), forward_name,
                    len(backward_logs), backward_name))
    return forward_logs, backward_logs


def parse_json_config(args):
    if not os.path.exists(args.input_json_file):
        raise ValueError(
            "The input_json_file {} is not found. Please check the path of json file.".
            format(args.input_json_file))

    origin_configs = []
    if os.path.isdir(args.input_json_file):
        if args.similar_api:
            for api in args.similar_api:
                json_path = os.path.join(args.input_json_file, api + '.json')
                with open(json_path, 'r') as f:
                    api_configs = json.load(f)
                    origin_configs.extend(api_configs)
        else:
            raise ValueError(
                "When input_json_file is a Directory, the args similar_api should be set."
            )
    elif os.path.isfile(args.input_json_file):
        with open(args.input_json_file, 'r') as f:
            origin_configs = json.load(f)

    configs_without_input, all_shapes, input_type = get_input_shapes_from_json(
        args, origin_configs)
    print("Total config of input json: {}".format(len(origin_configs)))

    return origin_configs, configs_without_input, all_shapes, input_type


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--op_name', type=str, default=None, help='Specify the operator name.')
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='Specify the path of log file.')
    parser.add_argument(
        '--input_json_file',
        type=str,
        default=None,
        required=True,
        help='Specify the path of input json file.')
    parser.add_argument(
        '--output_json_file',
        type=str,
        default=None,
        required=True,
        help='Specify the path of output json file.')
    parser.add_argument(
        '--input_shape',
        nargs='*',
        help='Specify the name of input shape. If None, ["input_shape"] will be used.'
    )
    parser.add_argument(
        '--ignored_params',
        nargs='*',
        help='Specify the ignored param list, the configs will be filtered according to the other params.'
    )
    parser.add_argument(
        '--parse_runtime_log',
        action='store_true',
        help='Whether selecting configs by parsing runtime log. '
        'Default: False, selecting configs by parsing json config.')
    parser.add_argument(
        '--similar_api',
        nargs='*',
        help='Specify the similar APIs, the output config will be '
        'selected from the json file of these APIs.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print("ignored_params: {0}.".format(args.ignored_params))
    if args.parse_runtime_log:
        forward_logs, backward_logs = get_logs(args.op_name, args.log_file)
        select_configs_by_log(args, forward_logs, backward_logs)
    else:
        origin_configs, configs_without_input, all_shapes, input_type = parse_json_config(
            args)
        select_configs_by_json(args, origin_configs, configs_without_input,
                               all_shapes, input_type)
