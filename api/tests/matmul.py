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


class PDMatmul(PaddleAPIBenchmarkBase):
    def build_program(self, config):
        with fluid.program_guard(self.main_program, self.startup_program):
            x = fluid.data(
                name='x',
                shape=config.x_shape,
                dtype=config.x_dtype,
                lod_level=0)
            y = fluid.data(
                name='y',
                shape=config.y_shape,
                dtype=config.y_dtype,
                lod_level=0)
            x.stop_gradient = False
            y.stop_gradient = False
            result = fluid.layers.matmul(
                x=x,
                y=y,
                transpose_x=config.transpose_x,
                transpose_y=config.transpose_y,
                alpha=config.alpha)

            self.feed_vars = [x, y]
            self.fetch_vars = [result]
            if config.backward:
                self.append_gradients(result, [x, y])


class TFMatmul(TensorflowAPIBenchmarkBase):
    def build_graph(self, config):
        x = self.placeholder(
            name='x', shape=config.x_shape, dtype=config.x_dtype)
        y = self.placeholder(
            name='y', shape=config.y_shape, dtype=config.y_dtype)
        result = tf.matmul(
            a=x,
            b=y,
            transpose_a=config.transpose_x,
            transpose_b=config.transpose_y,
            adjoint_a=False,
            adjoint_b=False,
            a_is_sparse=False,
            b_is_sparse=False)

        self.feed_list = [x, y]
        self.fetch_list = [result]
        if config.backward:
            self.append_gradients(result, [x, y])


def register_api():
    REGISTER_API_INFO['matmul'] = ['matmul', 'matmul.json']


if __name__ == '__main__':
    test_main(PDMatmul(), TFMatmul(), config=APIConfig("matmul"))
