#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import os, sys
import importlib

package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(package_path)

from common.main import parse_args, is_paddle_enabled, is_tensorflow_enabled, is_torch_enabled, test_main, test_main_without_json
from common.registry import benchmark_registry


def import_tests(filename=None):
    special_module_list = ["__init__", "common_import", "test_main"]

    def _import_api(test_module_name, basename):
        try:
            module = importlib.import_module(test_module_name + "." + basename)
            print("-- Import {} successfully.".format(module.__name__))
            return module
        except Exception as e:
            print("-- Failed to import {}: {}".format(basename, e))
            return None

    test_module_name = "tests"
    if filename is not None:
        filenames = [filename]
    else:
        tests_path = os.path.join(package_path, test_module_name)
        filenames = sorted(os.listdir(tests_path))
    for f in filenames:
        api_name = os.path.splitext(f)[0]
        file_extension = os.path.splitext(f)[1]
        if file_extension == '.py' and api_name not in special_module_list:
            _import_api(test_module_name, api_name)


def main():
    args = parse_args()
    assert args.filename is not None, "Argument filename is not set."
    current_dir = os.path.dirname(os.path.abspath(__file__))
    abs_filepath = os.path.join(current_dir, args.filename + ".py")
    assert os.path.exists(abs_filepath), "{} is not exist.".format(
        abs_filepath)

    import_tests(filename=args.filename + ".py")
    print(benchmark_registry)

    info = benchmark_registry.get(args.filename)
    if info.op_type is not None:
        config = info.config_class(info.op_type)
    else:
        config = info.config_class()

    pd_obj = None
    tf_obj = None
    pd_dy_obj = None
    torch_obj = None
    if args.testing_mode == "dynamic":
        if is_paddle_enabled(args, config):
            pd_dy_obj = info.paddle_class("dynamic")
        if is_torch_enabled(args, config):
            torch_obj = info.pytorch_class()
    elif args.testing_mode == "static":
        if is_paddle_enabled(args, config):
            pd_obj = info.paddle_class("static")
        if is_tensorflow_enabled(args, config):
            tf_obj = info.tensorflow_class()
    print("-- pd_obj:", pd_obj)
    print("-- tf_obj:", tf_obj)
    print("-- pd_dy_obj:", pd_dy_obj)
    print("-- torch_obj:", torch_obj)

    test_main(
        pd_obj=pd_obj,
        tf_obj=tf_obj,
        pd_dy_obj=pd_dy_obj,
        torch_obj=torch_obj,
        config=config)


if __name__ == '__main__':
    main()
