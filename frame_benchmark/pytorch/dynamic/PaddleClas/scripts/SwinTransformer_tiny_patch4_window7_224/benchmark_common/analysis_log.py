import os
import argparse
import json
import re
from numpy import mean,var


def parse_args():
    def str2bool(v):
        if v.lower() in ('true', 't', '1'):
            return True
        else:
            return False

    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log', type=str, default='log path')
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('-b', '--batch_size', type=int, required=True)
    parser.add_argument('-n', '--device_num', type=str, required=True)
    parser.add_argument('-s', '--save_path', type=str, default=None)
    parser.add_argument('-f', '--fp', type=str, default='fp32')
    parser.add_argument('--skip_steps', type=int, default=0, help='The number of steps to be skipped')
    args = parser.parse_args()
    return args


def get_log_file(log_path):
    with open(log_path) as fd:
        lines = fd.readlines()
    return list(filter(lambda l: "Train:" in l, lines))


def calculate_ips(log_list, batch_size, skip_steps=0):
    pattern = re.compile(r"^.*Time:\s?(\d+\.?\d*)s,.*$")
    records = []
    for line in log_list:
        time = pattern.findall(line)[0]
        records.append(float(time))

    if len(records) < skip_steps + 10:
        print('ERROR!!! too few logs printed')
        return 0

    # skip后去掉去除前max(5%,5)和后max(5%,5)个数据再计算平均值
    sorted_records = sorted(records[skip_steps:])
    skip_step2 = max(int(len(sorted_records)*0.05), 5)
    try:
        del sorted_records[:skip_step2]
        del sorted_records[-skip_step2:]
        avg_time = mean(sorted_records)
    except Exception:
        print("no records")
        return 0
    ips = batch_size / avg_time
    return ips


if __name__ == "__main__":
    args = parse_args()
    try:
        num_gpu = int(args.device_num[3:])
        log_list = get_log_file(args.log)
        ips = calculate_ips(log_list, num_gpu * args.batch_size, args.skip_steps)
    except Exception as e:
        ips = 0

    run_mode = 'DP'
    if args.save_path:
        save_file = args.save_path
    else:
        save_file = 'clas_{}_{}_bs{}_fp32_{}_speed'.format(
            args.model_name, run_mode, args.batch_size, args.device_num)
    save_content = {
        "model_branch": os.getenv('model_branch'),
        "model_commit": os.getenv('model_commit'),
        "model_name": args.model_name+"_bs"+str(args.batch_size)+"_"+args.fp+"_"+run_mode,
        "batch_size": args.batch_size,
        "fp_item": args.fp,
        "run_mode": run_mode,
        "convergence_value": 0,
        "convergence_key": "",
        "ips": ips,
        "speed_unit": "images/s",
        "device_num": args.device_num,
        "model_run_time": os.getenv('model_run_time'),
        "frame_commit": "",
        "frame_version": os.getenv('frame_version'),
            }
    print(save_content)
    with open(save_file, 'w') as fd:
        json.dump(save_content, fd)
