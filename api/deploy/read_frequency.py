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


def read_frequency_from_text(op_frequency_path):
    op_frequency_dict = {}
    with open(op_frequency_path, "r") as f:
        for line in f.readlines():
            contents = line.split()
            if len(contents) != 3:
                continue
            op_frequency_dict[contents[1]] = int(contents[2])
    return op_frequency_dict
