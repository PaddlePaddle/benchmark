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
        if l.startswith('Epoch') and 'Batch' in l and int(
                l.split('Batch [')[1].split(']')[0]) > 50:
            log_list.append(
                float(l.split('Speed:')[1].split('samples')[0].strip()))

    return log_list


def calculate_ips(log_list, gpu_num):
    ips = 0
    for x in log_list:
        ips += x
    avg_ips = ips / len(log_list)
    return avg_ips


if __name__ == "__main__":
    args = parse_args()
    try:
        log_list = get_log_file(args.file)
        ips = calculate_ips(log_list, args.num_gpu)
    except Exception as e:
        ips = 0

    run_mode = 'sp' if args.num_gpu == 1 else 'mp'
    save_file = 'clas_{}_{}_bs{}_fp32_{}_speed'.format(args.model_name,
                                                       run_mode,
                                                       args.batch_size,
                                                       args.num_gpu)
    save_content = {}
    save_content["log_file"] = args.file
    save_content["model_name"] = "clas_{}_bs{}_fp32".format(
        args.model_name, args.batch_size)
    save_content["mission_name"] = "图像分类"
    save_content["direction_id"] = 0
    save_content["run_mode"] = run_mode
    save_content["index"] = 1
    save_content["gpu_num"] = args.num_gpu
    save_content["FILNAL_RESULT"] = ips
    save_content["JOB_FAIL_FLAG"] = 0 if ips > 0 else 1

    with open(save_file, 'w') as fd:
        json.dump(save_content, fd)
