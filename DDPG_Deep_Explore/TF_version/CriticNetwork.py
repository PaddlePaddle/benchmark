import tensorflow as tf
from tensorpack.utils import logger
from tensorpack.utils.globvars import globalns as param
from tf_utils import Fun, Model, global_norml_clip_wrapper
from tensorpack import *

HIDDEN1_UNITS = 800
HIDDEN2_UNITS = 400
VEL_HIDDEN1_UNITS = 200
VEL_HIDDEN2_UNITS = 400

class CriticNetwork(object):
    def __init__(self, sess, TAU, model_id, inputs):
        self.sess = sess
        self.TAU = TAU
        self.model_id = model_id
        
        # build inputs
        self._state, self._action, self._reward, self._done, self._next_state, self._lr_critic, self._lr_actor = inputs

        #Now create the model
        self.model, self.weights = self.create_critic_network(self._state, self._action, target=False) 
        ema = tf.train.ExponentialMovingAverage(decay=1 - self.TAU)
        def ema_getter(getter, name, *args, **kwargs):
          return ema.average(getter(name, *args, **kwargs))
        with tf.variable_scope("critic_moving_average", reuse=tf.AUTO_REUSE):
          self.target_update = ema.apply(self.weights)
        self.ema_getter = ema_getter
        self.ema = ema

    def combine_actor(self, actor):
      logger.info("[combine_actor] start")
      self.target_model, _ = self.create_critic_network(self._next_state, actor.target_model.output, target=True, custom_getter=self.ema_getter)
      target_q_values = self.target_model.output
      target_q_values = tf.where(self._done, self._reward, self._reward + param.gamma * target_q_values)
      target_q_values = tf.stop_gradient(target_q_values)

      #self.loss = tf.losses.huber_loss(target_q_values, self.model.output, delta=20.0, reduction=tf.losses.Reduction.MEAN)
      self.loss = tf.losses.mean_squared_error(target_q_values, self.model.output, reduction=tf.losses.Reduction.MEAN)

      with tf.variable_scope('critic_optimizer_{}'.format(self.model_id)):
        optimizer = tf.train.AdamOptimizer(self._lr_critic)
        grads = optimizer.compute_gradients(self.loss, self.weights)
        grads = global_norml_clip_wrapper(grads)
        #with tf.control_dependencies([self.target_update]):
        self.train_op = optimizer.apply_gradients(grads)
      logger.info("[CriticNetwork/{}] build done".format(self.model_id))

    def target_train(self):
        self.sess.run(self.target_train_op)

    def create_critic_network(self, state, action, target, custom_getter=None):
      if target:
        trainable = False
        assert custom_getter is not None
      else:
        assert custom_getter is None
        trainable = True

      real_state = state[:, :param.state_dim - param.vel_dim]
      vel = state[:, -param.vel_dim:]

      with argscope(FullyConnected, activation=tf.nn.selu, trainable=trainable):
        #with tf.variable_scope('critic_shared', reuse=tf.AUTO_REUSE, custom_getter=custom_getter):
        with tf.variable_scope('critic_identity_{}'.format(self.model_id), reuse=tf.AUTO_REUSE, custom_getter=custom_getter):
          w1 = FullyConnected('w1', real_state, HIDDEN1_UNITS)
          a1 = FullyConnected('a1', action, HIDDEN2_UNITS)
          h1 = FullyConnected('h1', w1, HIDDEN2_UNITS)
          vel_h0 = FullyConnected('vel_h0', vel, VEL_HIDDEN1_UNITS)
          vel_h1 = FullyConnected('vel_h1', vel_h0, VEL_HIDDEN2_UNITS)
          h2 = tf.concat([h1, a1, vel_h1], axis=-1)
        with tf.variable_scope('critic_identity_{}'.format(self.model_id), reuse=tf.AUTO_REUSE, custom_getter=custom_getter):
          h3 = FullyConnected('h3', h2, HIDDEN2_UNITS)
          V = FullyConnected('value', h3, 1, activation=None)
          V = tf.squeeze(V, axis=-1)
      weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_shared')
      weights += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic_identity_{}'.format(self.model_id))

      model = Model()
      model.predict = Fun([state, action], V, self.sess)
      model.output = V
      return model, weights
