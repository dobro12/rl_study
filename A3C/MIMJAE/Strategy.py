# -*- coding: utf-8 -*-

import pickle
import random
import collections
import numpy as np
import matplotlib.pyplot as plt

import gym
from Agent import Agent
from Q_Network import Q_Network
from utils import *

import subprocess
import sys
GITPATH = subprocess.run('git rev-parse --show-toplevel'.split(' '), \
        stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n','')
sys.path.append(GITPATH)
import dobroEnv

GAMMA = 0.99

class A3C:
    def __init__(self, seed = 0, agent_num = 5, act_num = 11, is_continuous = True,
                 lr_A = 1e-3, lr_V = 1e-3, file_name = 'test', Game = 'MountainCar-v0'):
        
        self.env = gym.make(Game)
        if Game == 'DobroHalfCheetah-v0':
            self.env.unwrapped.initialize(is_render=False)
        self.env.seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        rng = np.random.RandomState(seed)
        
        self.obs_dim = self.env.observation_space.shape[0]
        if is_continuous:
            self.act_dim = self.env.action_space.shape[0]
        else:
            self.act_dim = 1
        
        self.Game, self.seed = Game, seed
        
        if is_continuous:
            self.action_set = discretization_action_space(self.env, self.act_dim, act_num)
        else:
            self.action_set = [0, 1, 2]
        self.act_num = len(self.action_set)
        
        self.file_name = file_name
        self.agent_num = agent_num
        self.Central_Network = Q_Network(seed, self.obs_dim, self.act_dim, self.act_num, lr_A, lr_V)
        self.Central_Network.save_network(file_name)
        
        self.agent_list = []
        for i in range(agent_num):
            agent0 = Agent(seed, i, act_num, is_continuous, lr_A, lr_V, Game)
            agent0.load_central2local(file_name)
            self.agent_list.append(agent0)
        
        print("")
        print("Hi! I'm Minjae Kang")
        print("This is a A3C algorithm")
    
    def run_A3C(self, render=False, save_epi=100, save_network=False):
        MAX_epi, MAX_step, TOTAL_step, EPI_step = int(5e3), int(3e6), 0, 0
        if self.Game == 'DobroHalfCheetah-v0':
            MAX_step = int(1e7)
        Lens, Rews, Avg_Rews, ALoss_Epi, VLoss_Epi = [], [], [], [], []
        
        print("")
        print(self.Game +" CASE - SEED "+str(self.seed))
        print("  STATE DIM : {}, ACTION DIM : {}".format(self.obs_dim, self.act_dim))
        print("  Number of Actions : {}".format(len(self.action_set)))
        print("")
        
        episode = 0
        while TOTAL_step < MAX_step:
            which_agent = np.random.randint(self.agent_num)
            epi, done, steps, reward, a_loss, v_loss, a_gradients, v_gradients = self.agent_list[which_agent].do_step()
            TOTAL_step += 1
            
            if done:
                self.Central_Network.apply_a_gradients(a_gradients)
                self.Central_Network.apply_v_gradients(v_gradients)
                
                episode += 1
                EPI_step += steps
                Rews.append(reward)
                Lens.append(EPI_step)
                ALoss_Epi.append(a_loss)
                VLoss_Epi.append(v_loss)

                rew_avg = np.average(Rews[max(0, episode-100):])
                Avg_Rews.append(rew_avg)
                
                print()
                print("Agent {} completes {}th episode.".format(which_agent+1, epi))
                print("{}   {}".format(episode, round(Avg_Rews[episode-1],3)))
                print("            Result : {},  ALoss : {},  VLoss : {},  Steps : {},  Total Steps : {} "
                                    .format(round(Rews[-1], 3), round(float(ALoss_Epi[episode-1]), 3), round(float(VLoss_Epi[episode-1]), 3), steps, Lens[-1]))
                
                self.agent_list[which_agent].reset_env()
                self.Central_Network.save_network(self.file_name)
                self.agent_list[which_agent].load_central2local(self.file_name)
        
                if episode % save_epi == 0 and episode:
                    plt.figure(figsize=(15,4))
                    x_values = Lens[:]
                    y_values = Avg_Rews[:]
                    plt.subplot(131)
                    plt.plot(x_values, y_values, c='blue')
                    plt.ticklabel_format(axis='x', style='sci', scilimits=(5,5))
                    plt.title(self.file_name+'_Reward')
                    plt.grid(True)

                    z_values = ALoss_Epi[:]
                    plt.subplot(132)
                    plt.plot(x_values, z_values, c='orange')
                    plt.ticklabel_format(axis='x', style='sci', scilimits=(5,5))
                    plt.title(self.file_name+'_ActorLoss')
                    plt.grid(True)
                    
                    w_values = VLoss_Epi[:]
                    plt.subplot(133)
                    plt.plot(x_values, w_values, c='yellow')
                    plt.ticklabel_format(axis='x', style='sci', scilimits=(5,5))
                    plt.title(self.file_name+'_CriticLoss')
                    plt.grid(True)
                    plt.show()
                
            if TOTAL_step > MAX_step:
                plt.figure(figsize=(15,4))
                x_values = Lens[:]
                y_values = Avg_Rews[:]
                plt.subplot(131)
                plt.plot(x_values, y_values, c='blue')
                plt.ticklabel_format(axis='x', style='sci', scilimits=(5,5))
                plt.title(self.file_name+'_Reward')
                plt.grid(True)

                z_values = ALoss_Epi[:]
                plt.subplot(132)
                plt.plot(x_values, z_values, c='orange')
                plt.ticklabel_format(axis='x', style='sci', scilimits=(5,5))
                plt.title(self.file_name+'_ActorLoss')
                plt.grid(True)

                w_values = VLoss_Epi[:]
                plt.subplot(133)
                plt.plot(x_values, w_values, c='yellow')
                plt.ticklabel_format(axis='x', style='sci', scilimits=(5,5))
                plt.title(self.file_name+'_CriticLoss')
                plt.grid(True)
                plt.show()
                
                break
