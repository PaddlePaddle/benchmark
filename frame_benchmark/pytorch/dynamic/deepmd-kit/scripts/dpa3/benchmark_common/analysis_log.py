import os
import argparse
import json
import re
from numpy import mean,var


def parse_args():
    """
    解析命令行参数。
    
    Args:
        无
    
    Returns:
        argparse.Namespace: 包含解析后参数的命名空间对象。
    
    """
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


def parse_ips(log_path):
    """
    从日志行中提取平均训练时间（秒/批次）
    
    Args:
        log_line (str): 日志行文本
    
    Returns:
        Union[float, None]: 如果找到了匹配项，则返回平均训练时间（秒/批次），
        否则返回 None。
    
    """
    # 读取日志文件，只过滤出包含"average training time:"的行
    with open(log_path) as fd:
        lines = fd.readlines()
    for line in lines:
        if "average training time:" in line:
            match = re.search(r'average training time: ([\d\.]+) s/batch', line)
            print(line, float(match.group(1)))
            if match:
                return float(match.group(1))
    return 0


if __name__ == "__main__":
    args = parse_args()
    try:
        num_gpu = int(args.device_num[3:])
        ips = parse_ips(args.log)
    except Exception as e:
        ips = 0
    if args.save_path:
        save_file = args.save_path
    else:
        save_file = 'clas_{}_{}_bs{}_fp32_{}_speed'.format(
            args.model_name, run_mode, args.batch_size, args.device_num)
    save_content = {
        "model_branch": os.getenv('model_branch'),
        "model_commit": os.getenv('model_commit'),
        "model_name": args.model_name + "_bs" + str(args.batch_size) + "_" + args.fp + "_DP",
        "batch_size": args.batch_size,
        "fp_item": args.fp,
        "run_mode": 'DP',
        "convergence_value": 0,
        "convergence_key": "",
        "ips": ips,
        "speed_unit": "s/batch",
        "device_num": args.device_num,
        "model_run_time": os.getenv('model_run_time'),
        "frame_commit": "",
        "frame_version": os.getenv('frame_version'),
            }
    print(save_content)
    with open(save_file, 'w') as fd:
        json.dump(save_content, fd)