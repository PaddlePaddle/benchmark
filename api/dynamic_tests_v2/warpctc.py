#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from common_import import *

class PaddleWarpctc(PaddleDynamicAPIBenchmarkBase):
    def build_graph(self, config):
        logits_length = self.variable(name='logits_length', shape=config.logits_length_shape, 
                                      dtype=config.logits_length_dtype, 
                                      value=np.array([config.max_seq_length]*config.batch_size).astype("int64"))
        label_length = self.variable(name='label_length', shape=config.label_length_shape, 
                                     dtype=config.label_length_dtype, 
                                     value=np.array([config.max_label_length]*config.batch_size).astype("int64"))
        logits = self.variable(name='logits', shape=config.logits_shape, dtype=config.logits_dtype)
        label = self.variable(name='label', shape=config.label_shape, dtype=config.label_dtype)
        result = paddle.fluid.layers.warpctc(input=logits, label=label, input_length=logits_length, 
                                             label_length=label_length, blank=config.blank, norm_by_times=config.norm_by_times)
        self.feed_list = [logits_length, label_length, logits, label]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [logits])

if __name__ == '__main__':
    test_main(
        pd_dy_obj=PaddleWarpctc(),
        config=APIConfig('warpctc'))
