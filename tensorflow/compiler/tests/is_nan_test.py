# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.stack_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

from tensorflow.compiler.tests.xla_test import XLATestCase
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import tf_logging as logging

GPU_DEVICE = "/job:localhost/replica:0/task:0/device:GPU:0"

class IsNanTest(XLATestCase):

  def testPositive(self):
    m = 1024
    k = 1024
    shape0 = [m, k]
    dtype = dtypes.float32

    arg1_val = (np.random.random([m,k]) - 0.5 ) * 0.08
    arg2_val = (np.random.random([m,k]) - 0.5 ) * 0.04
    arg3_val = (np.random.random([m,k]) - 0.5 ) * 0.08
    arg4_val = (np.random.random([m,k]) - 0.5 ) * 0.04
    arg2_val[1000, 1000] = arg1_val[1000, 1000]
    arg3_val[1000, 1000] = arg4_val[1000, 1000]

    with self.session() as session:
      with self.test_scope():
        arg1 = array_ops.placeholder(dtype=dtype, shape=shape0, name="arg1")
        arg2 = array_ops.placeholder(dtype=dtype, shape=shape0, name="arg2")
        arg3 = array_ops.placeholder(dtype=dtype, shape=shape0, name="arg3")
        arg4 = array_ops.placeholder(dtype=dtype, shape=shape0, name="arg4")
        with ops.device(GPU_DEVICE):
          x = (arg1 - arg2) / (arg3 - arg4)
          result = math_ops.reduce_any(math_ops.is_nan(x))
      result_val = result.eval(feed_dict={arg1: arg1_val,
                                          arg2: arg2_val,
                                          arg3: arg3_val,
                                          arg4: arg4_val})
      self.assertTrue(result_val)

  def testNegative(self):
    m = 1024
    k = 1024
    shape0 = [m, k]
    dtype = dtypes.float32

    arg1_val = (np.random.random([m,k]) - 0.5 ) * 0.08
    arg2_val = (np.random.random([m,k]) - 0.5 ) * 0.04
    arg3_val = (np.random.random([m,k]) - 0.5 ) * 0.08
    arg4_val = (np.random.random([m,k]) - 0.5 ) * 0.04

    with self.session() as session:
      with self.test_scope():
        arg1 = array_ops.placeholder(dtype=dtype, shape=shape0, name="arg1")
        arg2 = array_ops.placeholder(dtype=dtype, shape=shape0, name="arg2")
        arg3 = array_ops.placeholder(dtype=dtype, shape=shape0, name="arg3")
        arg4 = array_ops.placeholder(dtype=dtype, shape=shape0, name="arg4")
        with ops.device(GPU_DEVICE):
          x = (arg1 - arg2) / (arg3 - arg4)
          result = math_ops.reduce_any(math_ops.is_nan(x))
      result_val = result.eval(feed_dict={arg1: arg1_val,
                                            arg2: arg2_val,
                                            arg3: arg3_val,
                                            arg4: arg4_val})
      self.assertFalse(result_val)
    
if __name__ == "__main__":
  googletest.main()
