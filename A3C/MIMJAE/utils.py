# -*- coding: utf-8 -*-
import numpy as np

GAMMA = 0.99

def discretization_action_space(env, dim_action, num_action):
    action_set = np.zeros((np.power(num_action, dim_action), dim_action))
    for i in range(dim_action):
        act_low  = env.action_space.low[0]
        act_high = env.action_space.high[0]
        act_tmp  = np.linspace(act_low, act_high, num=num_action)
        
        for j in range(num_action):
            idx_ = pow(num_action, i) * j
            while idx_ < len(action_set):
                for k in range(pow(num_action, i)):
                    action_set[idx_+k, i] = act_tmp[j]
                idx_ += np.power(num_action, i+1)
    return action_set

def convert_reward_with_current_policy(reward_batch, done_batch):
    reward_batch = np.flip(reward_batch)
    len_epi = len(reward_batch)
    new_reward_batch = []
    if done_batch[-1]:
        new_reward_batch.append(0.0)
    else: 
        new_reward_batch.append(reward_batch[0])
    for i in range(1, len_epi):
        new_reward_batch.append(reward_batch[i]+GAMMA*new_reward_batch[-1])
    new_reward_batch = np.flip(new_reward_batch[1:])
    return new_reward_batch  
