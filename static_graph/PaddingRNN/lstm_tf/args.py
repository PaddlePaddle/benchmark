#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_type",
        type=str,
        default="small",
        help="model_type [test|small|medium|large]")
#    parser.add_argument(
#        "--embed_size",
#        type=int,
#        default=300,
#        help="The dimension of embedding table. (default: %(default)d)")
#    parser.add_argument(
#        "--hidden_size",
#        type=int,
#        default=300,
#        help="The size of rnn hidden unit. (default: %(default)d)")
#    parser.add_argument(
#        "--batch_size",
#        type=int,
#        default=32,
#        help="The sequence number of a mini-batch data. (default: %(default)d)")
    parser.add_argument(
        "--max_epoch",
        type=int,
        default=0,
        help="The pass number to train. (default: %(default)d)")
#    parser.add_argument(
#        "--learning_rate",
#        type=float,
#        default=0.001,
#        help="Learning rate used to train the model. (default: %(default)f)")
#    parser.add_argument(
#        "--use_gpu",
#        type=distutils.util.strtobool,
#        default=True,
#        help="Whether to use gpu. (default: %(default)d)")
#    parser.add_argument(
#        "--debug",
#        action='store_true',
#        help="Whether to print debug info. (default: %(default)d)")
    parser.add_argument(
        "--profile",
        type=bool,
        default=False,
        help='Early exit if profile = True. (default: %(default)d)')
#    parser.add_argument(
#        "--save_dir",
#        type=str,
#        default="model",
#        help="Specify the path to save trained models.")
    parser.add_argument(
        "--inference_only",
        type=bool,
        default=False,
        help="if use inference only")

    parser.add_argument(
        "--rnn_type",
        type=str,
        default='static',
        help="which rnn type to use")

#    parser.add_argument(
#        "--save_interval",
#        type=int,
#        default=1,
#        help="Save the trained model every n passes."
#        "(default: %(default)d)")
#    parser.add_argument('--trainset', nargs='+', help='train dataset')
#    parser.add_argument('--devset', nargs='+', help='dev dataset')
#    parser.add_argument('--testset', nargs='+', help='test dataset')
#    parser.add_argument('--vocab_dir', help='dict')
#    parser.add_argument('--max_p_num', type=int, default=5)
#    parser.add_argument('--max_a_len', type=int, default=200)
#    parser.add_argument('--max_p_len', type=int, default=500)
#    parser.add_argument('--max_q_len', type=int, default=9)
#    parser.add_argument('--doc_num', type=int, default=1)
#    parser.add_argument('--single_doc', action='store_true')
#    parser.add_argument('--simple_net', type=int, default=0)
#    parser.add_argument('--para_init', action='store_true')
    parser.add_argument('--log_path', help='path of the log file. If not set, logs are printed to console')
    args = parser.parse_args()
    return args
