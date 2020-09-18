import argparse
import os
import sys
import subprocess
import numpy as np


def parse_args():
    """
    Parse the args of command line
    :return: all the args that user defined
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--start_commit',
        type=str,
        default='8ec4af278d59d58ed9acc4657c7b6126fcfd8992',
        help='the start pr of searching, it is a correct pr.'
    )
    parser.add_argument(
        '--end_commit',
        type=str,
        default='4adac0e30978f3a28d10adf30f319e96ee3e03ed',
        help='the end pr of searching, it is a pr with problem.'
    )
    parser.add_argument(
        '--is_perf',
        type=bool,
        default=False,
        help='find pr with performance problem or function problem, default is function problem'
    )
    parser.add_argument(
        '--command',
        type=str,
        default='CUDA_VISIBLE_DEVICES=7 bash run_benchmark.sh 1 sp 300',
        help='command of running the script'
    )
    parser.add_argument(
        '--standard_value',
        type=float,
        default=65.906,
        help='the standard value, only needed when is_perf=True'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.05,
        help='the threshold of alarming, default is 0.05'
    )
    args = parser.parse_args()
    return args


def compile(commit_id):
    uninstall_command = 'pip uninstall paddlepaddle-gpu -y'
    os.system(uninstall_command)
    reset_command = 'git reset --hard %s' % commit_id
    os.system(reset_command)
    command = 'rm -rf build && mkdir build'
    os.system(command)
    build_path = os.path.join(paddle_path, 'build')
    os.chdir(build_path)
    print('commit {} build start'.format(commit_id))
    cmake_command = 'cmake .. -DPY_VERSION=3.7 -DWITH_GPU=ON -DWITH_DISTRIBUTE=OFF -DWITH_TESTING=OFF -DWITH_INFERENCE_API_TEST=OFF -DON_INFER=OFF -DCMAKE_BUILD_TYPE=Release'
    build_command = 'make -j12'
    os.system(cmake_command)
    build_res = os.system(build_command)
    if build_res == 0:
        print('commit {} build done'.format(commit_id))
    else:
        print('commit {} build failed'.format(commit_id))
        return -1
    install_command = 'pip install -U python/dist/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl'
    os.system(install_command)
    print('commit {} install done'.format(commit_id))
    return 0


def check_success(commit_id):
    # os.chdir(paddle_path)
    # compile(commit_id)
    os.chdir(base_path)
    print('base_path:{}'.format(base_path))
    if not args.is_perf:
        cmd = args.command
        if os.system(cmd) == 0:
            return True
        else:
            return False
    else:
        cmd = args.command
        log = subprocess.getstatusoutput(cmd)
        print('log:{}'.format(log[1]))
        f = open('log.txt', 'w')
        f.writelines(log[1])
        f.close()
        cmd = 'cat log.txt | tr "," "\n" | grep FINAL_RESULT'
        log_result = subprocess.getstatusoutput(cmd)
        f = open('log_result.txt', 'w')
        f.writelines(log_result[1])
        f.close()
        with open('log_result.txt', 'r') as f:
            for line in f:
                sub_str = line.split(':')
                result = float(sub_str[1].strip('\n'))
                print('result is {}'.format(result))
                standard_value = args.standard_value
                print('standard_value is {}'.format(standard_value))
                if abs((result - standard_value) / standard_value) <= args.threshold:
                    return True
                else:
                    return False


def get_commits(start, end):
    print('start:{}'.format(start))
    print('end:{}'.format(end))
    cmd = 'git log {}..{} --pretty=oneline'.format(start, end)
    log = subprocess.getstatusoutput(cmd)
    print(log[1])
    commit_list = []
    candidate_commit = log[1].split('\n')
    print(candidate_commit)
    for commit in candidate_commit:
        commit = commit.split(' ')[0]
        print('commit:{}'.format(commit))
        commit_list.append(commit)
    return commit_list


def binary_search(commits):
    if len(commits) == 2:
        print('only two candidate commits left in binary_search, the final commit is {}'.format(commits[0]))
        return commits[0]
    left, right = 0, len(commits) - 1
    if left <= right:
        mid = left + (right - left) // 2
        commit = commits[mid]
        if check_success(commit):
            print('the commit {} is success'.format(commit))
            # right = mid
            print('mid value:{}'.format(mid))
            selected_commits = commits[:mid + 1]
            binary_search(selected_commits)
        else:
            print('the commit {} is failed'.format(commit))
            # left = mid
            selected_commits = commits[mid:]
            binary_search(selected_commits)


if __name__ == '__main__':
    args = parse_args()
    base_path = os.getcwd()
    paddle_path = os.path.join(base_path, 'paddle')
    cmd = 'rm -rf paddle && git clone http://github.com/paddlepaddle/paddle.git'
    os.system(cmd)
    os.chdir(paddle_path)
    commits = get_commits(start=args.start_commit, end=args.end_commit)
    print('the candidate commits is {}'.format(commits))
    final_commit = binary_search(commits)
    print('the pr with problem is {}'.format(final_commit))
    f = open('final_commit.txt', 'w')
    f.writelines('the final commit is:{}'.format(final_commit))
    f.close()
