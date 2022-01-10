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

import abc, six


@six.add_metaclass(abc.ABCMeta)
class BenchmarkBase(object):
    def __init__(self, testing_mode):
        self.name = self.__class__.__name__
        self.feed_list = None
        self.fetch_list = None
        self._backward = False
        self._testing_mode = testing_mode

    @property
    def backward(self):
        return self._backward

    def compute_flop_and_byte(self, config):
        """ flop is used as a metric for op's performance and it is optional.
        """
        return None, None

    @abc.abstractmethod
    def build_graph(self, config=None):
        pass

    @abc.abstractmethod
    def variable(self, name, shape, dtype, value=None, stop_gradient=False):
        pass

    @abc.abstractmethod
    def layers(self, api_name, module_name=None, **kwargs):
        pass

    @abc.abstractmethod
    def append_gradients(self, targets, inputs):
        pass
