import os
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True)
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('-b', '--batch_size', type=int, required=True)
    parser.add_argument('-n', '--device_num', type=str, required=True)
    parser.add_argument('-s', '--save_path', type=str, default=None)
    parser.add_argument('--fp', type=str, default='fp32')
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
        num_gpu = int(args.device_num[3:])
        log_list = get_log_file(args.file)
        ips = calculate_ips(log_list, num_gpu)
    except Exception as e:
        ips = 0

    run_mode = 'sp' if args.num_gpu == 1 else 'mp'
    save_file = 'clas_{}_{}_bs{}_fp32_{}_speed'.format(args.model_name,
                                                       run_mode,
                                                       args.batch_size,
                                                       args.device_num)
    run_mode = 'SP' if num_gpu == 1 else 'MP'
    if args.save_path:
        save_file = args.save_path
    else:
        save_file = 'clas_{}_{}_bs{}_fp32_{}_speed'.format(
            args.model_name, run_mode, args.batch_size, args.device_num)
    save_content = {
        "model_branch": os.getenv('model_branch'),
        "model_commit": os.getenv('model_commit'),
        "model_name": args.model_name+"_bs"+str(args.batch_size)+"_"+args.fp+run_mode,
        "batch_size": args.batch_size,
        "fp_item": args.fp,
        "run_process_type": "MultiP",
        "run_mode": run_mode,
        "convergence_value": 0,
        "convergence_key": "",
        "ips": ips,
        "speed_unit": "images/s",
        "device_num": args.device_num,
        "model_run_time": os.getenv('model_run_time'),
        "frame_commit": "",
        "frame_commit": "",
            }
    print(save_content)
    with open(save_file, 'w') as fd:
        json.dump(save_content, fd)
