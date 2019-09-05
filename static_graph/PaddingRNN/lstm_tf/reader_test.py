# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

"""Tests for models.tutorials.rnn.ptb.reader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path


import reader




def testPtbProducer():
    raw_data = [4, 3, 2, 1, 0, 5, 6, 1, 1, 1, 1, 0, 3, 4, 1]
    batch_size = 3
    num_steps = 2
    iter = reader.get_data_iter(raw_data, batch_size, num_steps)
    
    print( iter )
    for batch in iter:
        x,y = batch
        print( x )
        print( y )

        print( "11")
    
    '''
    hidden_size = 200
        self.assertAllEqual(xval, [[4, 3], [5, 6], [1, 0]])
        self.assertAllEqual(yval, [[3, 2], [6, 1], [0, 3]])
        xval, yval = session.run([x, y])
        self.assertAllEqual(xval, [[2, 1], [1, 1], [3, 4]])
        self.assertAllEqual(yval, [[1, 0], [1, 1], [4, 1]])
      finally:
        coord.request_stop()
        coord.join()
    '''
testPtbProducer()

