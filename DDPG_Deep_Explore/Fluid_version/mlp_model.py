#-*- coding: utf-8 -*-
#File: actor_model.py

import sys
sys.path.append('../PARL/')

import parl.layers as layers
from paddle import fluid
from paddle.fluid.param_attr import ParamAttr
from parl.framework.model_base import Model
import numpy as np

class MLPModel(Model):
    def __init__(self, obs_dim, vel_obs_dim, act_dim, model_id=-1, shared=False):
        super(MLPModel, self).__init__()
        self.actor_model = ActorModel(obs_dim, vel_obs_dim, act_dim, model_id, shared)
        self.critic_model = CriticModel(obs_dim, vel_obs_dim, act_dim, model_id, shared)

    def policy(self, obs):
        return self.actor_model.predict(obs)

    def value(self, obs, action):
        return self.critic_model.predict(obs, action)

    def policy_parameters(self):
        return self.actor_model.parameter_names

class ActorModel(Model):
    def __init__(self, obs_dim, vel_obs_dim, act_dim, model_id=-1, shared=False):
        super(ActorModel, self).__init__()
        hid0_size = 800
        hid1_size = 400
        hid2_size = 200
        vel_hid0_size = 200
        vel_hid1_size = 400

        self.obs_dim = obs_dim
        self.vel_obs_dim = vel_obs_dim

        # actor network layers
        if shared:
            scope_name = 'policy_shared'
        else:
            scope_name = 'policy_identity_{}'.format(model_id)
        
        # buttom layers
        self.fc0 = layers.fc(size=hid0_size,
                             act='tanh',
                             param_attr=ParamAttr(name='{}/h0/W'.format(scope_name)),
                             bias_attr=ParamAttr(name='{}/h0/b'.format(scope_name)))
        self.fc1 = layers.fc(size=hid1_size,
                             act='tanh',
                             param_attr=ParamAttr(name='{}/h1/W'.format(scope_name)),
                             bias_attr=ParamAttr(name='{}/h1/b'.format(scope_name)))
        self.vel_fc0 = layers.fc(size=vel_hid0_size,
                                 act='tanh',
                                 param_attr=ParamAttr(name='{}/vel_h0/W'.format(scope_name)),
                                 bias_attr=ParamAttr(name='{}/vel_h0/b'.format(scope_name)))
        self.vel_fc1 = layers.fc(size=vel_hid1_size,
                                 act='tanh',
                                 param_attr=ParamAttr(name='{}/vel_h1/W'.format(scope_name)),
                                 bias_attr=ParamAttr(name='{}/vel_h1/b'.format(scope_name)))

        # top layers
        self.fc2 = layers.fc(size=hid2_size,
                             act='tanh',
                             param_attr=ParamAttr(name='policy_identity_{}/h2/W'.format(model_id)),
                             bias_attr=ParamAttr(name='policy_identity_{}/h2/b'.format(model_id)))
        self.fc3 = layers.fc(size=act_dim,
                             act='tanh',
                             param_attr=ParamAttr(name='policy_identity_{}/means/W'.format(model_id)),
                             bias_attr=ParamAttr(name='policy_identity_{}/means/b'.format(model_id)))

    def predict(self, obs):
        real_obs = layers.slice(obs, axes=[1], starts=[0], ends=[self.obs_dim - self.vel_obs_dim])
        vel_obs = layers.slice(obs, axes=[1], starts=[-self.vel_obs_dim], ends=[self.obs_dim])
        hid0 = self.fc0(real_obs)
        hid1 = self.fc1(hid0)
        vel_hid0 = self.vel_fc0(vel_obs)
        vel_hid1 = self.vel_fc1(vel_hid0)
        concat = layers.concat([hid1, vel_hid1], axis=1)
        hid2 = self.fc2(concat)
        means = self.fc3(hid2) 
        return means

class CriticModel(Model):
    def __init__(self, obs_dim, vel_obs_dim, act_dim, model_id=0, shared=False):
        super(CriticModel, self).__init__()
        hid0_size = 800
        hid1_size = 400
        vel_hid0_size = 200
        vel_hid1_size = 400

        self.obs_dim = obs_dim
        self.vel_obs_dim = vel_obs_dim

        # actor network layers
        if shared:
            scope_name = 'critic_shared'
        else:
            scope_name = 'critic_identity_{}'.format(model_id)
        
        # buttom layers
        self.fc0 = layers.fc(size=hid0_size,
                             act='relu',
                             param_attr=ParamAttr(name='{}/w1/W'.format(scope_name)),
                             bias_attr=ParamAttr(name='{}/w1/b'.format(scope_name)))
        self.fc1 = layers.fc(size=hid1_size,
                             act='relu',
                             param_attr=ParamAttr(name='{}/h1/W'.format(scope_name)),
                             bias_attr=ParamAttr(name='{}/h1/b'.format(scope_name)))
        self.vel_fc0 = layers.fc(size=vel_hid0_size,
                                 act='relu',
                                 param_attr=ParamAttr(name='{}/vel_h0/W'.format(scope_name)),
                                 bias_attr=ParamAttr(name='{}/vel_h0/b'.format(scope_name)))
        self.vel_fc1 = layers.fc(size=vel_hid1_size,
                                 act='relu',
                                 param_attr=ParamAttr(name='{}/vel_h1/W'.format(scope_name)),
                                 bias_attr=ParamAttr(name='{}/vel_h1/b'.format(scope_name)))
        self.act_fc0 = layers.fc(size=hid1_size,
                                 act='relu',
                                 param_attr=ParamAttr(name='{}/a1/W'.format(scope_name)),
                                 bias_attr=ParamAttr(name='{}/a1/b'.format(scope_name)))

        # top layers
        self.fc2 = layers.fc(size=hid1_size,
                             act='relu',
                             param_attr=ParamAttr(name='critic_identity_{}/h3/W'.format(model_id)),
                             bias_attr=ParamAttr(name='critic_identity_{}/h3/b'.format(model_id)))
        self.fc3 = layers.fc(size=1,
                             act='relu',
                             param_attr=ParamAttr(name='critic_identity_{}/value/W'.format(model_id)),
                             bias_attr=ParamAttr(name='critic_identity_{}/value/b'.format(model_id)))

    def predict(self, obs, action):
        real_obs = layers.slice(obs, axes=[1], starts=[0], ends=[self.obs_dim - self.vel_obs_dim])
        vel_obs = layers.slice(obs, axes=[1], starts=[-self.vel_obs_dim], ends=[self.obs_dim])
        hid0 = self.fc0(real_obs)
        hid1 = self.fc1(hid0)
        vel_hid0 = self.vel_fc0(vel_obs)
        vel_hid1 = self.vel_fc1(vel_hid0)
        a1 = self.act_fc0(action)
        concat = layers.concat([hid1, a1, vel_hid1], axis=1)
        hid2 = self.fc2(concat)
        V = self.fc3(hid2)
        V = layers.squeeze(V, axes=[1])
        return V
