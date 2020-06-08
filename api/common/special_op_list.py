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

RANDOM_OP_LIST = ["dropout"]

NO_FETCHES_OPS = ["feed", "null"]

# operators without grad ops.
NO_BACKWARD_OPS = [
    "accuracy", "argsort", "assign", "cast", "compare", "less_than",
    "less_equal", "not_equal", "greater_than", "greater_equal", "equal",
    "cumsum", "feed", "fetch", "fill_constant", "increment", "isfinite",
    "logical_not", "logical_and", "logical_or", "null", "one_hot", "scale",
    "sequence_mask", "shape", "zeros_like", "instance_norm"
]

# When running the API backwards in HANG_CASES, the progress will hang.
HANG_CASES = ["instance_norm"]

NO_NEED_ARGS = {
    "batch_norm": ["moving_mean_name", "moving_variance_name"],
    "embedding": ["is_distributed"]
}

CONTROL_FLOW_OPS = [
    "conditional_block", "switch", "static_rnn", "while", "while_loop", "cond",
    "case", "ifelse", "dynamic_rnn", "switch_case"
]

EXCLUDE_OPS = [
    "create_tensor", "create_parameter", "create_global_var",
    "autoincreased_step_counter"
]

ALIAS_OP_MAP = {
    "arg_max": "argmax",
    "arg_min": "argmin",
    "nearest_interp": "resize_nearest",
    "bilinear_interp": "resize_bilinear",
    "depthwise_conv2d": "conv2d",
    "sum": "sums",
    "hierarchical_sigmoid": "hsigmoid",
    "sample_logits": "sampled_softmax_with_cross_entropy"
}
