import argparse
import random
import os
import json
import warnings


def select_configs(args, forward_logs, backward_logs):
    removed_params = ["op"]
    if args.ignored_params:
        removed_params += args.ignored_params
    combined_logs = combine_logs_with_key_params(removed_params, forward_logs,
                                                 backward_logs)
    config_groups = grouping_configs(combined_logs)
    print("==================config_groups===================")
    num_configs = args.num_configs
    if not num_configs:
        num_configs = len(config_groups)
    selected_config_ids = []
    for key in config_groups:
        print("config: {0}, total: {1}".format(key, len(config_groups[key])))
        config_ids = config_groups[key]
        number = max(1, num_configs * len(config_ids) / len(combined_logs))
        ids = random.sample(config_ids, int(number))
        print("Select {0} config_ids: {1}.".format(number, ids))
        selected_config_ids += ids

    with open(args.json_file, 'r') as f:
        all_configs = json.load(f)
    out_dir = os.path.dirname(args.output_file)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    configs = []
    for index in selected_config_ids:
        configs.append(all_configs[index])
    with open(args.output_file, 'w') as f:
        json.dump(configs, f, indent=4, sort_keys=True)


def combine_logs_with_key_params(removed_params, forward_logs, backward_logs):
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


def grouping_configs(logs):
    config_groups = dict()
    for i in range(len(logs)):
        if logs[i] not in config_groups.keys():
            config_groups[logs[i]] = [i]
        else:
            config_groups[logs[i]] += [i]
    return config_groups


def remove_params(log, removed_params):
    result = list(log)
    if removed_params:
        for item in log:
            param_val = item.split("=")
            if param_val[0] in removed_params:
                result.remove(item)

    return result


def get_logs(op_name, log_file):
    forward_logs = []
    backward_logs = []
    forward_name = "op=" + op_name
    backward_name = forward_name + '_grad'
    with open(log_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if backward_name in line:
                index = line.index(backward_name)
                line = line[index:].replace(', ', ',').split(' ')
                backward_logs.append(line)
            elif forward_name in line:
                index = line.index(forward_name)
                line = line[index:].replace(', ', ',').split(' ')
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--op_name',
        type=str,
        default=None,
        required=True,
        help='Specify the operator name.')
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        required=True,
        help='Specify the path of log file.')
    parser.add_argument(
        '--json_file',
        type=str,
        default=None,
        required=True,
        help='Specify the path of json file.')
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        required=True,
        help='Specify the path of json file.')
    parser.add_argument(
        '--ignored_params',
        nargs='*',
        help='Specify the ignored param list, the configs will be filtered according to the other params.'
    )
    parser.add_argument(
        '--num_configs',
        type=int,
        default=None,
        help='Specify the maximum number of selected configs.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print("ignored_params: {0}.".format(args.ignored_params))
    forward_logs, backward_logs = get_logs(args.op_name, args.log_file)
    select_configs(args, forward_logs, backward_logs)
