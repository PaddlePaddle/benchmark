#-*- coding: utf-8 -*-
#File: tf_utils.py
#Author: yobobobo(zhouboacmer@qq.com)

import tensorflow as tf

class Fun:
  """ Creates a python function that maps between inputs and outputs in the computational graph. """

  def __init__(self, inputs, outputs, session=None):
    self._inputs = inputs if type(inputs) == list else [inputs]
    self._outputs = outputs
    self._session = session or tf.get_default_session()

  def __call__(self, *args, **kwargs):
    """
    rewrite Fun()
    """

    feeds = {}
    for (argpos, arg) in enumerate(args):
      feeds[self._inputs[argpos]] = arg

    out = self._outputs
    res = self._session.run(out, feeds)

    return res

def global_norml_clip_wrapper(grads):
  g = [k[0] for k in grads]
  v = [k[1] for k in grads]
  g, _ = tf.clip_by_global_norm(g, 5.0, name='clip_by_global_norm')
  return list(zip(g, v))


class Model(object):
  """fake model class for TF to keep same API as Keras"""
  def __init__(self):
    pass

CURRENT_VARS = []
def record_vars():
  global CURRENT_VARS
  CURRENT_VARS = tf.trainable_variables()  

def get_new_vars():
  global CURRENT_VARS
  now_vars = tf.trainable_variables()
  ret_vars = []
  for var in now_vars:
    if var not in CURRENT_VARS:
      ret_vars.append(var)
  return ret_vars
