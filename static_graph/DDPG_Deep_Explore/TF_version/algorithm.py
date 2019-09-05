#-*- coding: utf-8 -*-
#File: algorithm.py
#Author: yobobobo(zhouboacmer@qq.com)

from tensorpack.utils.globvars import globalns as param
import tensorflow as tf
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
import numpy as np
from tensorpack.utils import logger
import threading as th
import queue
import time
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ensemble_num', type=int)
parser.add_argument('--test_times', type=int, default=10)
args = parser.parse_args()

param.ensemble_num = args.ensemble_num
param.action_dim = 19
param.vel_dim = 4
param.state_dim = 185 + param.vel_dim
param.gamma = 0.96

class Algorithm(object):
  config = tf.ConfigProto()
  config.inter_op_parallelism_threads = 1
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  TAU = 0.001
  lr_actor = 3e-5
  lr_critic = 3e-5
  train_times = 100
  BATCH_SIZE = 128
  ensemble_num = param.ensemble_num

  inputs =  [tf.placeholder(tf.float32, (None, param.state_dim), 'state'),
             tf.placeholder(tf.float32, (None, param.action_dim), 'action'),
             tf.placeholder(tf.float32, (None, ), 'reward'),
             tf.placeholder(tf.bool, (None, ), 'done'),
             tf.placeholder(tf.float32, (None, param.state_dim), 'next_state'),
             tf.placeholder(tf.float32, (), 'lr_critic'),
             tf.placeholder(tf.float32, (), 'lr_actor')]

  actors = []
  critics = []
  locks = []
  for idx in range(ensemble_num):
    actors.append(ActorNetwork(sess, TAU, idx, inputs))
    critics.append(CriticNetwork(sess, TAU, idx, inputs))
    locks.append(th.Lock())
  for i in range(ensemble_num):
    critics[i].combine_actor(actors[i])
    actors[i].combine_critic(critics[i])
  sess.run(tf.global_variables_initializer())

  def __init__(self):
    self.global_step = 0
    self.max_reward = 0
    self.max_shaping_reward = 0
    self.max_r2_reward = 0
    self.max_recent_reward = 0

    varlist = tf.trainable_variables()
    self.saver = tf.train.Saver(varlist, max_to_keep=0)
  
  def learn(self):
    result_q = queue.Queue()
    th_list = []
    for i in range(self.ensemble_num):
      t = th.Thread(target=self.train_single_model, args=(i, result_q))
      t.start()
      th_list.append(t)

    for t in th_list:
      t.join()
    for _ in range(self.ensemble_num):
      result = result_q.get()

  def train_single_model(self, model_idx, result_q):
    logger.info("[train_single_model] model_idx:{}".format(model_idx))
    critic_loss_list = []
    lock = self.locks[model_idx]
    actor = self.actors[model_idx]
    lr_critic = self.lr_critic * (1.0 + 0.1 * model_idx)
    lr_actor = self.lr_actor * (1.0 - 0.05 * model_idx)
    states = np.random.random((self.BATCH_SIZE, param.state_dim))
    actions = np.random.random((self.BATCH_SIZE, param.action_dim))
    rewards = np.random.random(self.BATCH_SIZE)
    dones = np.array([False] * self.BATCH_SIZE, dtype='bool')
    new_states = np.random.random((self.BATCH_SIZE, param.state_dim))
    for T in range(self.train_times):
      lock.acquire()
      _, _, critic_loss = actor.combine_train(states,actions,rewards,dones,new_states, lr_critic, lr_actor)
      lock.release()
      critic_loss_list.append(critic_loss)
    result_q.put(critic_loss_list)

if __name__ == '__main__':
  
  T = args.test_times

  alg = Algorithm()
  start_time = time.time()
  for i in tqdm(range(T)):
    alg.learn()
  logger.info("[learn] {} heads, time consuming:{}".format(alg.ensemble_num, (time.time() - start_time) / T))
