#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import os
import subprocess


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def run_command(command, shell=True):
    print("run command: %s" % command)
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=shell)

    exit_code = None
    stdout = ''
    while exit_code is None or line:
        exit_code = p.poll()
        line = p.stdout.readline().decode('utf-8')
        stdout += line

    return stdout, exit_code


def check_commit():
    try:
        import tensorflow as tf
        tf_version = tf.__version__
    except Exception:
        tf_version = None

    try:
        import torch
        torch_version = torch.__version__
    except Exception:
        torch_version = None

    try:
        current_dir = os.getcwd()
        print("-- Current directory: %s" % current_dir)

        dir_of_this_file = os.path.dirname(os.path.abspath(__file__))
        print("-- Entering %s" % dir_of_this_file)
        os.chdir(dir_of_this_file)
        print("-- Current directory: %s" % os.getcwd())
        benchmark_commit, _ = run_command("git rev-parse HEAD")
        benchmark_commit = benchmark_commit.replace("\n", "")
        benchmark_update_time, _ = run_command("git show -s --format=%ad")
        benchmark_update_time = benchmark_update_time.replace("\n", "")
        os.chdir(current_dir)
        print("-- Current directory: %s" % os.getcwd())

        import paddle
        paddle_version = paddle.version.full_version
        paddle_commit = paddle.version.commit

        print(
            "==========================================================================="
        )
        print("-- paddle version             : %s" % paddle_version)
        print("-- paddle commit              : %s" % paddle_commit)

        if tf_version:
            print("-- tensorflow version         : %s" % tf_version)

        if torch_version:
            print("-- pytorch version            : %s" % torch_version)

        print("-- benchmark commit           : %s" % benchmark_commit)
        print("-- benchmark last update time : %s" % benchmark_update_time)
        print(
            "==========================================================================="
        )
    except Exception:
        pass
