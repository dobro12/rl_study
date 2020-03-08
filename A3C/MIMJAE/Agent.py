# -*- coding: utf-8 -*-

import pickle
import random
import collections
import numpy as np
import matplotlib.pyplot as plt

import gym
from Q_Network import Q_Network
from utils import *

import subprocess
import sys
GITPATH = subprocess.run('git rev-parse --show-toplevel'.split(' '), \
        stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n','')
sys.path.append(GITPATH)
import dobroEnv

GAMMA = 0.99

class Agent:
    def __init__(self, seed = 0, agent_idx = 0, act_num = 11, is_continuous = True,
                 lr_A = 1e-3, lr_V = 1e-3, Game = 'MountainCar-v0'):
        
        self.env = gym.make(Game)
        if Game == 'DobroHalfCheetah-v0':
            self.env.unwrapped.initialize(is_render=False)
        self.env.seed(seed+agent_idx)
        np.random.seed(seed)
        random.seed(seed)
        rng = np.random.RandomState(seed)
        
        self.obs_dim = self.env.observation_space.shape[0]
        if is_continuous:
            self.act_dim = self.env.action_space.shape[0]
        else:
            self.act_dim = 1
        self.max_episode_steps = self.env._max_episode_steps
        
        self.Game, self.seed = Game, seed
        
        if is_continuous:
            self.action_set = discretization_action_space(self.env, self.act_dim, act_num)
        else:
            self.action_set = [0, 1, 2]
        self.act_num = len(self.action_set)
        
        self.Q_Network = Q_Network(seed+agent_idx, self.obs_dim, self.act_dim, self.act_num, lr_A, lr_V)
        
        self.episode, self.done, self.steps, self.reward = 1, False, 0, 0
        self.state = self.env.reset()
        
        self.cur_state, self.cur_action, self.cur_reward, self.cur_done = [], [], [] ,[]
        self.cur_state.append(self.state)
        
    def do_step(self):
        action_probs = np.reshape(self.Q_Network.get_a(self.state), [-1])
        action = np.random.choice(len(action_probs), size=1, p=action_probs)[0]

        next_state, reward, done, info = self.env.step(self.action_set[action])
        if self.steps == self.max_episode_steps-1:
            done_ = False
        else:
            done_ = done
        
        self.done = done
        self.reward += reward
        self.steps += 1
        
        action_one = np.zeros(np.shape(action_probs))
        action_one[action] = 1
        
        self.cur_state.append(next_state)
        self.cur_action.append(action_one)
        self.cur_reward.append(reward)
        self.cur_done.append(done_)
        
        self.state = next_state
        
        a_gradients, v_gradients = [], []
        a_loss, v_loss = 0, 0
        if self.done:
            v_t_1_batch = np.reshape(self.Q_Network.get_v(self.cur_state), [-1])
            self.cur_reward.append(v_t_1_batch[-1])
            
            cur_state  = np.array(self.cur_state)
            cur_action = np.array(self.cur_action)
            cur_reward = np.array(self.cur_reward)
            cur_done   = np.array(self.cur_done)
            new_reward = convert_reward_with_current_policy(cur_reward, cur_done)
            new_reward = np.array(new_reward)
            
            td_batch = new_reward - v_t_1_batch[:-1]
            v_t_1_batch = cur_reward[:-1] + GAMMA*v_t_1_batch[1:]
            a_loss, a_gradients = self.Q_Network.get_a_gradients(cur_state[:-1], cur_action, td_batch)
            v_loss, v_gradients = self.Q_Network.get_v_gradients(cur_state[:-1], v_t_1_batch)
            a_loss = np.mean(a_loss)/self.steps
            v_loss = np.mean(v_loss)
        return self.episode, self.done, self.steps, self.reward, a_loss, v_loss, a_gradients, v_gradients
            
    def reset_env(self):
        self.episode += 1
        self.state = self.env.reset()
        self.done, self.steps, self.reward = False, 0, 0

        self.cur_state, self.cur_action, self.cur_reward, self.cur_done = [], [], [] ,[]
        self.cur_state.append(self.state)
        
    def load_central2local(self, _name):
        self.Q_Network.load_network(_name)
        
        
        