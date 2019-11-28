#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from api_benchmark import APIBenchmarkBase

from args import parse_args
import tensorflow as tf

class abs(APIBenchmarkBase):
    def build_graph(self, backward=False):
        data = tf.placeholder(name='data', shape=[10, 10, 100, 100], dtype=tf.float32)
        result = tf.abs(x=data)

        self.feed_list = [data]
        if backward:
            gradients = tf.gradients(result, [data])
            self.fetch_list = [result, gradients[0]]
        else:
            self.fetch_list = [result]


if __name__ == '__main__':
    args = parse_args()
    obj = abs()
    obj.build_graph(backward=args.backward)
    obj.run(use_gpu=args.use_gpu,
            repeat=args.repeat,
            log_level=args.log_level,
            check_output=args.check_output,
            profile=args.profile)
    
