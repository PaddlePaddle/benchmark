import argparse
import os
import sys
import numpy as np

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(base_path)
print sys.path
import models.benchmark_server.helper as helper
from benchmark_server import benchmark_models as bm


def parse_args():
    """
    Parse the args of command line
    :return: all the args that user defined
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--start_day',
        type=str,
        default='2020-01-01',
        help='start day of searching, the correct pr is found in this day'
    )
    parser.add_argument(
        '--end_day',
        type=str,
        default='2020-09-15',
        help='end day of searching, the wrong pr is found in this day'
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
        default='sh /ssd3/xiege/run_benchmark.sh',
        help='path of the running script'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='bert',
        help='model_name,only needed when is_perf=True'
    )
    parser.add_argument(
        '--run_machine_type',
        type=str,
        default='ONE_GPU',
        help='ONE_GPU|FOUR_GPU|MULTI_GPU|MULTI_GPU_MULTI_PROCESS,only needed when is_perf=True'
    )
    parser.add_argument(
        '--graph_type',
        type=str,
        default='static_graph',
        help='static_graph|dynamic_graph,only needed when is_perf=True'
    )
    args = parser.parse_args()
    return args


def compile(commit_id):
    reset_command = 'git reset --hard %s' % commit_id
    os.system(reset_command)
    command = 'rm -rf build && mkdir build'
    os.system(command)
    print('commit {} build start'.format(commit_id))
    cmake_command = 'cmake .. -DPY_VERSION=3.7 -DWITH_GPU=ON -DWITH_DISTRIBUTE=OFF -DWITH_TESTING=OFF -DWITH_INFERENCE_API_TEST=OFF -DON_INFER=OFF -DCMAKE_BUILD_TYPE=Release'
    build_command = 'make - j$(nproc)'
    os.system(cmake_command)
    os.system(build_command)
    print('commit {} build done'.format(commit_id))
    install_command = 'pip install python/dist/paddlepaddle_gpu-0.0.0-cp37-cp37m-linux_x86_64.whl'
    os.system(install_command)
    print('commit {} install done'.format(commit_id))


def get_standard_value(model, env):
    if args.graph_type == 'static_graph':
        job_type = 2
    else:
        job_type = 3
    benchmark_results = bm.ViewJobResult.objects.filter(model_name=args.model_name, report_index_id=1,
                                                        job_type=job_type, cuda_version=10.1, cudnn_version=7,
                                                        device_type='v100', model_implement_type=args.graph_type,
                                                        frame_name="paddlepaddle", run_machine_type=args.graph_type,
                                                        benchmark=1).order_by('-version')
    benchmark_results = [float(i.report_result) for i in benchmark_results[:1]] if benchmark_results else []
    benchmark_result = round(np.array(benchmark_results).mean(), 4)
    return benchmark_result


def check_success(commit_id):
    compile(commit_id)
    if not args.is_perf:
        cmd = args.command
        if os.system(cmd) == 0:
            return True
        else:
            return False
    else:
        cmd = args.command + '| tee run.log'
        os.system(cmd)
        cmd = 'cat run.log | tr "," "\n" | grep FINAL_RESULT | tee result.log'
        os.system(cmd)
        with open('result.log', 'r') as f:
            sub_str = f.split(':')
            result = float(sub_str[1].strip('\n'))
            standard_value = get_standard_value(args.model, args.env)
            if abs((result - standard_value) / standard_value) <= 0.5:
                return True
            else:
                return False


def get_commits(start, end):
    cmd = 'git log --pretty=oneline --after="{}" --before="{}" | tee git.log'.format(start, end)
    os.system(cmd)
    commit_list = []
    with open("git.log", 'r') as f:
        for line in f:
            commit = line.split()[0]
            commit_list.append(commit)
    return commit_list


def binary_search(commits):
    if len(commits) == 2:
        print('only two candidate commits left in binary_search, the final commit is {}'.format(commits[0]))
        return commits[0]
    left, right = 0, len(commits) - 1
    while left <= right:
        mid = left + (right - left) // 2
        commit = commits[mid]
        if check_success(commit):
            print('the commit {} is success'.format(commit))
            right = mid
            selected_commits = commits[:, mid + 1]
            binary_search(selected_commits)
        else:
            print('the commit {} is failed'.format(commit))
            left = mid
            selected_commits = commits[mid, :]
            binary_search(selected_commits)


if __name__ == '__main__':
    args = parse_args()
    cmd = 'git clone http://github.com/paddlepaddle/paddle.git && cd paddle'
    os.system(cmd)
    commits = get_commits(start=args.start_day + ' ' + '0:00:00', end=args.end_day + ' ' + '23:59:59')
    final_commit = binary_search(commits)
    print('the pr with problem is {}'.format(final_commit))
