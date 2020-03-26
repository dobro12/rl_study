# -*- coding: utf-8 -*-
import numpy as np

GAMMA = 0.99

def train_qnetwork(Q_Network, train_batch, input_size, num_actions):
    state_t_batch, state_t_1_batch, action_batch, reward_batch, done_batch = zip(*train_batch)
    
    state_t_batch   = np.array(state_t_batch)
    state_t_1_batch = np.array(state_t_1_batch)
    action_batch = np.array(action_batch)
    reward_batch = np.array(reward_batch)
    done_batch   = np.array(done_batch)
    
    batch_size = np.shape(state_t_batch)[0]
    
    action_t_1_batch = Q_Network.get_ta(state_t_1_batch)
    q_t_1_batch = Q_Network.get_tq(state_t_1_batch, action_t_1_batch)
    q_t_1_batch = np.reshape(q_t_1_batch, [-1])
    q_t_1_batch = reward_batch + GAMMA*q_t_1_batch*(1-done_batch)
    q_t_1_batch = np.reshape(q_t_1_batch, [-1,1])
    
    errors, cost, _ = Q_Network.train_qnetwork(state_t_batch, action_batch, q_t_1_batch)
    cost = np.mean(cost)
    return cost

def train_anetwork(Q_Network, train_batch, input_size, num_actions):
    state_t_batch, state_t_1_batch, action_batch, reward_batch, done_batch = zip(*train_batch)
    
    state_t_batch = np.array(state_t_batch)
    state_t_1_batch = np.array(state_t_1_batch)
    action_batch = np.array(action_batch)
    reward_batch = np.array(reward_batch)
    done_batch = np.array(done_batch)
    
    batch_size = np.shape(state_t_batch)[0]
    
    q_gradients = Q_Network.get_q_gradients(state_t_batch, action_batch)
    _ = Q_Network.train_anetwork(state_t_batch, q_gradients[0])

