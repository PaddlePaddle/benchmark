# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import argparse
import os
import json

import sys
sys.path.append("..")
from common import utils
from common import api_param


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--task',
        type=str,
        default="speed",
        help='Specify the task: [speed|accuracy]')
    parser.add_argument(
        '--framework',
        type=str,
        default="paddle",
        help='Specify the framework: [paddle|tensorflow|tf|both]')
    parser.add_argument(
        '--json_file', type=str, default=None, help='The file of API params')
    parser.add_argument(
        '--config_id',
        type=int,
        default=None,
        help='Only import params of API from json file in the specified position [0|1|...]'
    )
    parser.add_argument(
        '--check_output',
        type=str2bool,
        default=True,
        help='Whether checking the consistency of outputs [True|False]')
    parser.add_argument(
        '--profiler',
        type=str,
        default="none",
        help='Choose which profiler to use [\"none\"|\"Default\"|\"OpDetail\"|\"AllOpDetail\"|\"nvprof\"]'
    )
    parser.add_argument(
        '--backward',
        type=str2bool,
        default=False,
        help='Whether appending grad ops [True|False]')
    parser.add_argument(
        '--use_gpu',
        type=str2bool,
        default=False,
        help='Whether using gpu [True|False]')
    parser.add_argument(
        '--repeat', type=int, default=1, help='Iterations of Repeat running')
    parser.add_argument(
        '--log_level', type=int, default=0, help='level of logging')
    parser.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        help='GPU id when benchmarking for GPU')
    args = parser.parse_args()
    gpu_id = args.gpu_id if args.gpu_id > 0 else 0
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) is None:
        print("CUDA_VISIBLE_DEVICES is None, set to CUDA_VISIBLE_DEVICES={}".
              format(gpu_id))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if args.task not in ["speed", "accuracy"]:
        raise ValueError("task should be speed, accuracy")
    if args.framework not in ["paddle", "tensorflow", "tf", "both"]:
        raise ValueError("task should be paddle, tensorflow, tf, both")

    if args.task == "accuracy":
        args.repeat = 1
        args.log_level = 0
        args.check_output = False
        args.profiler = "none"
    return args


def test_main(pd_obj=None, tf_obj=None, config=None):
    assert config is not None, "API config must be set."

    args = parse_args()
    if args.json_file is not None:
        if args.config_id is not None and args.config_id >= 0:
            config.init_from_json(args.json_file, args.config_id)
            test_main_without_json(pd_obj, tf_obj, config)
        else:
            num_configs = 0
            with open(args.json_file, 'r') as f:
                num_configs = len(json.load(f))
            for config_id in range(0, num_configs):
                config.init_from_json(args.json_file, config_id)
                test_main_without_json(pd_obj, tf_obj, config)
    else:
        test_main_without_json(pd_obj, tf_obj, config)


def test_main_without_json(pd_obj, tf_obj=None, config=None):
    assert config is not None, "API config must be set."

    args = parse_args()
    config.backward = args.backward
    use_feed_fetch = False if args.task == "speed" else True

    feed_dict = pd_obj.generate_feed_dict(config)
    if args.task == "accuracy" or args.framework in ["paddle", "both"]:
        pd_outputs = pd_obj.run(config, args, use_feed_fetch, feed_dict)

    if args.task == "accuracy" or args.framework in [
            "tensorflow", "tf", "both"
    ]:
        assert tf_obj is not None, "TensorFlow object is None."
        tf_config = config.to_tensorflow()
        tf_outputs = tf_obj.run(tf_config, args, use_feed_fetch, feed_dict)

    if args.task == "accuracy":
        utils.check_outputs(pd_outputs, tf_outputs, name=config.name)
