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
        log_probs = self.variable(name='log_probs', shape=config.log_probs_shape, dtype=config.log_probs_dtype)
        labels = self.variable(name='labels', shape=config.labels_shape, dtype=config.labels_dtype)
        input_lengths = self.variable(name='input_lengths', shape=config.input_lengths_shape, 
                                      dtype=config.input_lengths_dtype, 
                                      value=np.array([config.max_seq_length]*config.batch_size).astype("int64"))
        label_lengths = self.variable(name='label_lengths', shape=config.label_lengths_shape, 
                                     dtype=config.label_lengths_dtype, 
                                     value=np.array([config.max_label_length]*config.batch_size).astype("int64"))

        result = paddle.nn.functional.ctc_loss(log_probs=log_probs, labels=labels, input_lengths=input_lengths, 
                                    label_lengths=label_lengths, blank=config.blank, 
                                    reduction=config.reduction, norm_by_times=config.norm_by_times)
        
        self.feed_list = [log_probs, labels, input_lengths, label_lengths]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [log_probs])

if __name__ == '__main__':
    test_main(
        pd_dy_obj=PaddleWarpctc(),
        config=APIConfig('warpctc'))
