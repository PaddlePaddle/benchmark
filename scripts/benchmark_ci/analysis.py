import argparse
import ast
import json
import os


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
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.05,
        help='threshold')
    args = parser.parse_args()
    return args


def traverse_logs():
    file_list = []
    for dirpath, dirnames, filenames in os.walk(args.log_path):

        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            file_list.append(file_path)

    print('file_list:{}'.format(file_list))
    return file_list


def analysis():
    file_list = traverse_logs()
    for file in file_list:
        with open(file, 'r') as f:
            lines = f.readlines()
            try:
                job_info = json.loads(lines[-1])
            except Exception as e:
                print("file {} analysis error".format(file))
            result = json.dumps(job_info["FINAL_RESULT"])
            model = json.dumps(job_info["model_name"])
            model = model.strip('"')
            print("result:{}".format(result))
            print("model:{}".format(model))
            standard_record = os.path.join(args.standard_path, model + '.txt')
            with open(standard_record, 'r') as f:
                for line in f:
                    standard_result = float(line.strip('\n'))
                    print("standard_result:{}".format(standard_result))
                    ranges = round((float(result) - standard_result) / standard_result, 4)
                    if ranges >= args.threshold:
                        os.environ['err_flag'] = True
                        print("{}, FAIL".format(model))
                        print(
                            "Model {}'s result is {}, standard value is {}, the increase is "
                            "larger than threshold, please contact xiege01 or heya02 to modify "
                            "the standard value".format(model, result, standard_result))
                    elif ranges <= -args.threshold:
                        os.environ['err_flag'] = True
                        print("{}, FAIL".format(model))
                        print(
                            "Model {}'s result is {}, standard value is {}, the decrease is "
                            "larger than threshold".format(model, result, standard_result))
                    else:
                        print("{}, SUCCESS".format(model))


if __name__ == '__main__':
    args = parse_args()
    analysis()
