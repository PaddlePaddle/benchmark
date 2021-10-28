import os
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='work_dirs')
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('-b', '--batch_size', type=int, required=True)
    parser.add_argument('-n', '--num_gpu', type=int, required=True)
    args = parser.parse_args()
    return args


def get_log_file(root, ends_with='json'):
    for r, d, f in os.walk(root):
        for ff in f:
            if ff.endswith(ends_with):
                log_path = os.path.join(r, ff)
                with open(log_path) as fd:
                    lines = fd.readlines()
                return lines[1:], log_path


def calculate_ips(log_list, batch_size):
    if len(log_list) < 5:
        print('log number is smaller than 5, the ips may be inaccurate!')
    else:
        log_list = log_list[4:]
    time = 0
    for x in log_list:
        time += float(eval(x.strip())["time"])
    avg_time = time / len(log_list)
    ips = batch_size / avg_time
    return ips


if __name__ == "__main__":
    args = parse_args()
    try:
        log_list, log_path = get_log_file(args.dir)
        ips = calculate_ips(log_list, args.num_gpu * args.batch_size)
    except Exception as e:
        ips = 0

    run_mode = 'sp' if args.num_gpu == 1 else 'mp'
    save_file = 'clas_{}_{}_bs{}_fp32_{}_speed'.format(args.model_name,
                                                       run_mode,
                                                       args.batch_size,
                                                       args.num_gpu)
    save_content = {}
    save_content["log_file"] = log_path
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
