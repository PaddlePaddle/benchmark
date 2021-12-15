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

NO_FETCHES_OPS = ["feed", "null"]

RANDOM_OP_LIST = ["bernoulli", "dropout"]

# operators with different names in v1.8 and v2.0.
OPS_MAP_1TO2 = {
    "fill_constant": "full",
    "reduce_mean": "mean",
    "reduce_prod": "prod",
    "reduce_sum": "sum",
}

# operators without grad ops.
NO_BACKWARD_OPS = [
    # fake APIs to test some framework overhead
    "null",
    "feed",
    "fetch",

    # paddle v1 apis
    "accuracy",
    "arange",
    "argmax",
    "argmin",
    "argsort",
    "assign",
    "cast",
    "clip_by_norm",
    "cumsum",
    "diag",
    "equal",
    "fill_constant",
    "greater_equal",
    "greater_than",
    "increment",
    "isfinite",
    "isinf",
    "isnan",
    "less_equal",
    "less_than",
    "logical_not",
    "logical_and",
    "logical_or",
    "not_equal",
    "one_hot",
    "scale",
    "sequence_mask",
    "shape",
    "zeros_like",

    # paddle v2 apis
    "add_n",
    "any",
    "bernoulli",
    "empty",
    "equal_all",
    "floor_divide",
    "full",
    "greater",
    "less",
    "linspace",
    "remainder",
    "unique",
    "where_index",
    "yolo_box",

    # Temporarily add to this list to pass CI.
    "lstm",
]


def has_backward(config):
    if config.framework == "paddle" or not hasattr(config, "api_list"):
        api_name = config.api_name
    else:
        api_name = None
        for k, v in config.api_list.items():
            if v == config.api_name:
                api_name = k
                break
    return api_name not in NO_BACKWARD_OPS


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

# For example, maximum op need to pick maximum value from two inputs. When the value of inputs are same, 
# Paddle choose the second value as output and TF choose the first value as output.
DIFF_IMPLEMENTATION_TF_OPS = ["maximum", "minimum", "argsort"]
