import os
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True)
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('-b', '--batch_size', type=int, required=True)
    parser.add_argument('-n', '--num_gpu', type=int, required=True)
    args = parser.parse_args()
    return args


def get_log_file(file_path):
    with open(file_path) as fd:
        lines = fd.readlines()
    log_list = []

    for l in lines:
        if l.startswith('Epoch:') and 'lr' in l and int(
                l.split("[")[2].split('/')[0]) > 10:
            log_list.append(float(
                l.split('time:')[1].split('data')[0].strip()))

    return log_list


def calculate_ips(log_list, batch_size):
    time = 0
    for x in log_list:
        time += x
    avg_time = time / len(log_list)
    ips = round((batch_size / avg_time), 3)
    return ips


if __name__ == "__main__":
    args = parse_args()
    try:
        log_list = get_log_file(args.file)
        ips = calculate_ips(log_list, args.num_gpu * args.batch_size)
    except Exception as e:
        ips = 0

    run_mode = 'sp' if args.num_gpu == 1 else 'mp'
    save_file = 'clas_{}_{}_bs{}_fp32_{}_speed'.format(args.model_name,
                                                       run_mode,
                                                       args.batch_size,
                                                       args.num_gpu)
    save_content = {}
    save_content["log_file"] = args.file
    save_content["model_name"] = "{}_bs{}_fp32".format(
        args.model_name, args.batch_size)
    save_content["mission_name"] = "图像分类"
    save_content["direction_id"] = 0
    save_content["run_mode"] = run_mode
    save_content["index"] = 1
    save_content["UNIT"] = "images/s"
    save_content["gpu_num"] = args.num_gpu
    save_content["FINAL_RESULT"] = ips
    save_content["JOB_FAIL_FLAG"] = 0 if ips > 0 else 1

    with open(save_file, 'w') as fd:
        json.dump(save_content, fd)
