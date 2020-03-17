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
CAPA_MEMORY = int(1e6)

class DDPG:
    def __init__(self, seed = 0, batch_size = 64, lr_Q = 1e-3, lr_A = 1e-3,
                 Game = 'Safexp-Point'):
        
        self.env = gym.make(Game)
        self.env.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        rng = np.random.RandomState(seed)
        
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.act_high = self.env.action_space.high[0]
        """
        for o in range(self.act_dim):
            print(self.env.action_space.low[o])
            print(self.env.action_space.high[o])
        """
        self.seed, self.Game, self.batch_size = seed, Game, batch_size
        self.Q_Network = Q_Network(seed, self.obs_dim, self.act_dim, self.act_high, batch_size, lr_A, lr_Q)
        
        print("")
        print("Hi Im Minjae Kang!")
        print("This is a DDPG algorithm.")
    
    def run_DDPG(self, file_name='test', render=False, train=True, save_epi=200, save_network=False):
        file_case = str(self.seed)
                        
        if self.Game == 'DobroHalfCheetah-v0':
            self.env.unwrapped.initialize(is_render=render)
        if train:
            MAX_epi, MAX_step, TOTAL_step = int(5e3), int(5e6), 0
        else:
            MAX_epi, MAX_step, TOTAL_step = int(1e2), int(5e6), 0
        
        Lens, Rews, Avg_Rews, Loss_Epi = [], [], [], []
        
        if not train:
            self.Q_Network.load_network(file_name)
        
        replay_buffer = collections.deque()
        
        print("")
        print(self.Game +" CASE - SEED "+str(self.seed))
        print("  STATE DIM : {}, ACTION DIM : {}".format(self.obs_dim, self.act_dim))
        print("")
        
        for episode in range(1, MAX_epi+1):
            done, steps, result, unsafe, qloss = False, 0, 0, 0, 0
            state = self.env.reset()
            
            while not done:
                if train:
                    start_step = 10000
                else:
                    start_step = 0
                
                if TOTAL_step >= start_step:
                    action = self.Q_Network.get_a(state)[0]
                    if train:
                        action += np.random.random(self.act_dim)*1e-3
                else:
                    action = 2*self.act_high*np.random.random(self.act_dim)-self.act_high
                #print(action)
                    
                next_state, reward, done, info = self.env.step(action)
                result, steps, TOTAL_step = result+reward, steps+1, TOTAL_step+1
                done_ = False if steps == self.env._max_episode_steps else done
                
                if train:
                    view_episode = 10
                else:
                    view_episode = 1
                    
                if render and episode % view_episode == 0:
                    self.env.render()
                    
                if train:
                    replay_buffer.append((state, next_state, action, reward, done_))
                    if len(replay_buffer) > CAPA_MEMORY:
                        replay_buffer.popleft()
                state = next_state
                
                if train and len(replay_buffer) > self.batch_size:
                    minibatch = random.sample(replay_buffer, self.batch_size)
                    train_anetwork(self.Q_Network, minibatch, self.obs_dim, self.act_dim)
                    Qloss = train_qnetwork(self.Q_Network, minibatch, self.obs_dim, self.act_dim)
                    qloss += Qloss

                    if TOTAL_step % 1 == 0:
                        self.Q_Network.update_target_anet()
                        self.Q_Network.update_target_qnet()
            
            Rews.append(result)
            Lens.append(TOTAL_step)
            Loss_Epi.append(qloss/steps)
            
            rew_avg = np.average(Rews[max(0, episode-50):])
            Avg_Rews.append(rew_avg)
            
            print("{}   {}".format(episode, round(Avg_Rews[episode-1],2)))
            print ("            Result : {},  Loss : {}, Steps : {},  Total Steps : {} "
                                .format(round(result, 3), round(float(Loss_Epi[episode-1]),4), steps, TOTAL_step))
                
            if save_network and train and episode % 200 == 0:
                print("* Reach the save time in episode {}.".format(episode))
                self.Q_Network.save_network(_name=file_name)
                
            if episode % save_epi == 0:
                np.savez('./results/'+file_name+'.npz',
                         rew_train = Avg_Rews, len_train = Lens)

                plt.figure(figsize=(10,3))
                x_values = list(range(1, episode+1))
                y_values = Avg_Rews[:]
                plt.subplot(121)
                plt.plot(x_values, y_values, c='hotpink')
                plt.title(file_name+'_Reward')
                plt.grid(True)
                
                y_values = Loss_Epi[:]
                plt.subplot(122)
                plt.plot(x_values, y_values, c='orange')
                plt.title(file_name+'_LossQ')
                plt.grid(True)
                plt.show()
                
            if TOTAL_step > MAX_step:
                break
                
        plt.figure(figsize=(10,3))
        x_values = list(range(1, episode+1))
        y_values = Avg_Rews[:]
        plt.subplot(121)
        plt.plot(x_values, y_values, c='hotpink')
        plt.title(file_name+'_Reward')
        plt.grid(True)

        y_values = Loss_Epi[:]
        plt.subplot(122)
        plt.plot(x_values, y_values, c='orange')
        plt.title(file_name+'_LossQ')
        plt.grid(True)
        plt.show()
