import tensorflow as tf
from tensorpack.utils import logger
from tensorpack.utils.globvars import globalns as param
from tensorpack import *
from tf_utils import Fun, Model, global_norml_clip_wrapper

HIDDEN1_UNITS = 800
HIDDEN2_UNITS = 400
HIDDEN3_UNITS = 200
VEL_HIDDEN1_UNITS = 200
VEL_HIDDEN2_UNITS = 400

class ActorNetwork(object):
    def __init__(self, sess, TAU, model_id, inputs):
        self.sess = sess
        self.TAU = TAU
        self.model = Model()
        self.model_id = model_id
        self.inputs = inputs

        # inputs
        self._state, self._action, self._reward, self._done, self._next_state, self._lr_critic, self._lr_actor = inputs
        
        #Now create the model
        self.model, self.weights = self.create_actor_network(self._state, target=False)

        ema = tf.train.ExponentialMovingAverage(decay=1 - self.TAU)
        def ema_getter(getter, name, *args, **kwargs):
          return ema.average(getter(name, *args, **kwargs))
        with tf.variable_scope("actor_moving_average", reuse=tf.AUTO_REUSE):
          self.target_update = ema.apply(self.weights)

        self.ema = ema

        self.target_model, self.target_weights = self.create_actor_network(self._next_state, target=True, custom_getter=ema_getter)

    def combine_critic(self, critic):
      a_for_grad = self.model.output
      critic_model_for_grad, _ = critic.create_critic_network(self._state, a_for_grad, target=False)
      target = -1.0 * tf.reduce_mean(critic_model_for_grad.output)
      with tf.variable_scope("actor_optimizer_{}".format(self.model_id)):
        optimizer = tf.train.AdamOptimizer(learning_rate=self._lr_actor)
        grads = optimizer.compute_gradients(target, self.weights)
        grads = global_norml_clip_wrapper(grads)
        #with tf.control_dependencies([self.target_update]): 
        self.train_op = optimizer.apply_gradients(grads)
      self.combine_train = Fun(self.inputs, [self.train_op, critic.train_op, critic.loss], self.sess)
      logger.info("[ActorNetwork/{}] build done".format(self.model_id))

    def create_actor_network(self, state, target, custom_getter=None):
      if target:
        trainable = False
        assert custom_getter is not None
      else:
        assert custom_getter is None
        trainable = True

      real_state = state[:, :param.state_dim - param.vel_dim]
      vel = state[:, -param.vel_dim:]

      with argscope(FullyConnected, activation=tf.nn.tanh, trainable=trainable):
        with tf.variable_scope('policy_identity_{}'.format(self.model_id), reuse=tf.AUTO_REUSE, custom_getter=custom_getter):
          
          h0 = FullyConnected('h0', real_state, HIDDEN1_UNITS)
          h1 = FullyConnected('h1', h0, HIDDEN2_UNITS)
          vel_h0 = FullyConnected('vel_h0', vel, VEL_HIDDEN1_UNITS)
          vel_h1 = FullyConnected('vel_h1', vel_h0, VEL_HIDDEN2_UNITS)
          concat_out = tf.concat([h1, vel_h1], axis=1)
        with tf.variable_scope('policy_identity_{}'.format(self.model_id), reuse=tf.AUTO_REUSE, custom_getter=custom_getter):
          h2 = FullyConnected('h2', concat_out, HIDDEN3_UNITS)
          means = FullyConnected('means', h2, param.action_dim)

      weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='policy_shared')
      weights += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='policy_identity_{}'.format(self.model_id))

      model = Model()
      model.predict = Fun([self._state], means, self.sess)
      model.output = means
      return model, weights
