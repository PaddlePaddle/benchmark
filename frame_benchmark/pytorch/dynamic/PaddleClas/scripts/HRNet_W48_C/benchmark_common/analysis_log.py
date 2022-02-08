import os
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='work_dirs')
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('-b', '--batch_size', type=int, required=True)
    parser.add_argument('-n', '--device_num', type=str, required=True)
    parser.add_argument('-s', '--save_path', type=str, default=None)
    parser.add_argument('-f', '--fp', type=str, default='fp32')
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
        num_gpu = int(args.device_num[3:])
        log_list, log_path = get_log_file(args.dir)
        ips = calculate_ips(log_list, num_gpu)
    except Exception as e:
        ips = 0

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
        "frame_version": os.getenv('frame_version'),
            }
    print(save_content)
    with open(save_file, 'w') as fd:
        json.dump(save_content, fd)
