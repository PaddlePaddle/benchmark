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

from common_import import *
import numpy as np


class CosSimConfig(APIConfig):
    def __init__(self):
        super(CosSimConfig, self).__init__('cos_sim')
        self.run_tf = False


class PDCosSim(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            X = fluid.data(
                name='X',
                shape=config.X_shape,
                dtype=config.X_dtype,
                lod_level=0)
            X.stop_gradient = False
            Y = fluid.data(
                name='Y',
                shape=config.Y_shape,
                dtype=config.Y_dtype,
                lod_level=0)
            Y.stop_gradient = False
            result = fluid.layers.cos_sim(X=X, Y=Y)

            self.feed_vars = [X, Y]
            self.fetch_vars = [result]
            if config.backward:
                self.append_gradients(result, [X, Y])


class TFCosSim(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        X = np.random.random(config.X_shape).astype(config.X_dtype)
        Y = np.random.random(config.Y_shape).astype(config.Y_dtype)
        result = tf.compat.v1.losses.cosine_distance(
            labels=X, predictions=Y, axis=-1, weights=1.0, scope=None)

        self.feed_list = [X, Y]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [X, Y])


def register_api():
    REGISTER_API_INFO['cos_sim'] = ['cos_sim', 'cos_sim.json']


if __name__ == '__main__':
    test_main(PDCosSim(), TFCosSim(), config=CosSimConfig())
