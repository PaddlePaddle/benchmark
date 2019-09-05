#!/usr/bin/env python
# coding=utf8
# File: DDPG_alg.py
import sys
sys.path.append('../PARL/')

import parl.layers as layers
from copy import deepcopy
from mlp_model import MLPModel
from paddle import fluid
from parl.framework.algorithm_base import Algorithm


class DDPGAlgorithm(Algorithm):
    """ Multi head version DDPG algorithm
    """
    def __init__(self, obs_dim, vel_obs_dim, act_dim, GAMMA, TAU, gpu_id, ensemble_num=1):
        self.obs_dim = obs_dim
        self.vel_obs_dim = vel_obs_dim
        self.act_dim = act_dim
        self.GAMMA, self.TAU = GAMMA, TAU
        self.gpu_id = gpu_id
        self.ensemble_num = ensemble_num
        
        self.models = []
        self.target_models = []
        
        for i in range(ensemble_num):
            model = MLPModel(obs_dim, vel_obs_dim, act_dim, model_id=i)
            target_model = deepcopy(model)
            self.models.append(model)
            self.target_models.append(target_model)

            print('model_id: {}'.format(i))
            print('model: {}'.format(model.parameter_names))
            print('target_model: {}'.format(target_model.parameter_names))
    
    def _ensemble_predict(self, obs):
        raise NotImplemented

    def actor_predict(self, obs, model_id=None):
        if model_id is not None:
            return self.models[model_id].policy(obs)
        else:
            return self._ensemble_predict(obs)    

    def learn(self, obs, action, reward, next_obs, terminal, actor_lr, critic_lr, model_id):
        self._actor_learn(obs, actor_lr, model_id)
        critic_loss = self._critic_learn(obs, action, reward, next_obs, terminal, critic_lr, model_id)
        return critic_loss

    def _actor_learn(self, obs, actor_lr, model_id):
        action = self.models[model_id].policy(obs)
        Q = self.models[model_id].value(obs, action)
        loss = layers.reduce_mean(-1.0 * Q)
        #loss = layers.reduce_mean(Q)
        #loss = layers.scale(loss, scale=-1.0)
        optimizer = fluid.optimizer.AdamOptimizer(actor_lr)
        #optimizer = fluid.optimizer.SGD(actor_lr)
        optimizer.minimize(loss, parameter_list=self.models[model_id].policy_parameters())

    def _critic_learn(self, obs, action, reward, next_obs, terminal, critic_lr, model_id):
        next_action = self.target_models[model_id].policy(next_obs)
        next_Q = self.target_models[model_id].value(next_obs, next_action)

        terminal = layers.cast(terminal, dtype='float32')
        target_Q = reward + (1.0 - terminal) * self.GAMMA * next_Q
        target_Q.stop_gradient = True

        Q = self.models[model_id].value(obs, action)
        loss = layers.square_error_cost(Q, target_Q)
        loss = layers.reduce_mean(loss)
        optimizer = fluid.optimizer.AdamOptimizer(critic_lr)
        #optimizer = fluid.optimizer.SGD(critic_lr)
        optimizer.minimize(loss)
        return loss

    def sync_target(self, model_id, first_sync=False, share_vars_parallel_executor=None):
        if first_sync:
            decay = 0
        else:
            decay = 1 - self.TAU
        # Todo: sync actor and critic in one progrom
        #self.models[model_id].sync_params_to(self.target_models[model_id], gpu_id=self.gpu_id, decay=decay,
        #        share_vars_parallel_executor=share_vars_parallel_executor)
