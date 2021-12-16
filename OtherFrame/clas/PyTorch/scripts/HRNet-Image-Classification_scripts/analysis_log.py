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


def get_log_file(root, ends_with='log'):
    speed = []
    for r, d, f in os.walk(root):
        for ff in f:
            if ff.endswith(ends_with):
                log_path = os.path.join(r, ff)
                with open(log_path) as fd:
                    lines = fd.readlines()

                for l in lines:
                    if 'Speed' in l:
                        speed.append(
                            float(
                                l.split('Speed')[1].split('samples')
                                [0].strip()))
                return speed, log_path


def calculate_ips(log_list, gpu_num):
    if len(log_list) < 5:
        print('log number is smaller than 5, the ips may be inaccurate!')
    else:
        log_list = log_list[4:]
    return sum(log_list) / len(log_list)


if __name__ == "__main__":
    args = parse_args()
    try:
        log_list, log_path = get_log_file(args.dir)
        ips = calculate_ips(log_list, args.num_gpu)
    except Exception as e:
        ips = 0

    run_mode = 'sp' if args.num_gpu == 1 else 'mp'
    save_file = 'clas_{}_{}_bs{}_fp32_{}_speed'.format(args.model_name,
                                                       run_mode,
                                                       args.batch_size,
                                                       args.num_gpu)
    save_content = {}
    save_content["log_file"] = log_path
    save_content["model_name"] = "{}_bs{}_fp32".format(
        args.model_name, args.batch_size)
    save_content["mission_name"] = "图像分类"
    save_content["direction_id"] = 0
    save_content["run_mode"] = run_mode
    save_content["index"] = 1
    save_content["UNIT"] = "images/s"
    save_content["gpu_num"] = args.num_gpu
    save_content["FILNAL_RESULT"] = ips
    save_content["JOB_FAIL_FLAG"] = 0 if ips > 0 else 1

    with open(save_file, 'w') as fd:
        json.dump(save_content, fd)
