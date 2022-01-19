import argparse
import ast
import json
import os

from analysis import traverse_logs, analysis


def parse_args():
    """
    Parse the args of command line
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log_path',
        type=str,
        default='../../logs/static',
        help='path of benchmark logs')
    parser.add_argument(
        '--standard_path',
        type=str,
        default='../../scripts/benchmark_ci/standard_value/static',
        help='path of standard_value')
    args = parser.parse_args()
    return args


def modify_standard_value():
    """
    Setting paddle develop result as standard value
    """
    file_list = traverse_logs(args.log_path)
    for file in file_list:
        model, fail_flag, result, loss_result = analysis(file)
        if int(fail_flag) == 1:
            print("{} running failed in paddle develop!".format(model))
        else:
            print("result:{}".format(result))
            print("model:{}".format(model))
            standard_record = os.path.join(args.standard_path, model + '.txt')
            with open(standard_record, 'r') as f:
                for line in f:
                    standard_result = line.strip('\n')
                    print("Setting paddle develop result as standard value.")
                    command = 'sed -i "s/{}/{}/g" {}'.format(standard_result, result, standard_record)
                    os.system(command)

if __name__ == '__main__':
    args = parse_args()
    modify_standard_value()
