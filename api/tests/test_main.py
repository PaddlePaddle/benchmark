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

import os
import op_benchmark_info

from common.main import parse_args, is_paddle_enabled, is_tensorflow_enabled, is_torch_enabled, test_main, test_main_without_json
from common.registry import benchmark_registry


def main():
    print(benchmark_registry)
    args = parse_args()
    assert args.filename is not None, "Argument filename is not set."
    abs_filepath = os.path.abspath(args.filename + ".py")
    assert os.path.exists(abs_filepath), "{} is not exist.".format(
        abs_filepath)

    info = benchmark_registry.get(args.filename)
    if info.op_type is not None:
        config = info.config_class(info.op_type)
    else:
        config = info.config_class()
    if args.testing_mode == "dynamic":
        pd_dy_obj = info.paddle_dynamic_class() if is_paddle_enabled(
            args, config) else None
        torch_obj = info.pytorch_class() if is_torch_enabled(args,
                                                             config) else None

    test_main(pd_dy_obj=pd_dy_obj, torch_obj=torch_obj, config=config)


if __name__ == '__main__':
    main()
