# to ignore warning message
import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
###########################
import subprocess
import sys
GITPATH = subprocess.run('git rev-parse --show-toplevel'.split(' '), \
        stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n','')
sys.path.append(GITPATH)
import dobroEnv
##############################################################

from graph_drawer import Graph
from logger import Logger
from nets import Agent

from collections import deque
import numpy as np
import pickle
import time
import sys
import gym

#env_name = 'DobroHalfCheetah-v0'
env_name = 'HalfCheetah-v0'

save_name = env_name.split('-')[0]
agent_args = {'agent_name':'TRPO',
            'env_name':save_name,
            'discount_factor':0.99,
            'hidden1':128,
            'hidden2':128,
            'v_lr':1e-3,
            'value_epochs':20,
            'num_conjugate':10,
            'max_decay_num':10,
            'line_decay':0.5,
            'max_kl':0.01,
            'std':0.1,
            }

def train():
    global env_name, save_name, agent_args
    env = gym.make(env_name)
    env.unwrapped.initialize(is_render=False)
    agent = Agent(env, agent_args)

    v_loss_logger = Logger(save_name, 'v_loss')
    p_loss_logger = Logger(save_name, 'p_loss')
    score_logger = Logger(save_name, 'score')
    graph = Graph(1000, save_name.upper(), agent.name)
    episodes = 10
    epochs = int(1e5)
    save_freq = 10

    save_period = 100
    p_losses = deque(maxlen=save_period)
    v_losses = deque(maxlen=save_period)
    entropies = deque(maxlen=save_period)
    scores = deque(maxlen=save_period*episodes)

    for epoch in range(epochs):
        states = []
        actions = []
        targets = []
        ep_step = 0
        for episode in range(episodes):
            state = env.reset()
            done = False
            score = 0
            step = 0
            temp_rewards = []
            while not done:
                step += 1
                ep_step += 1
                action, clipped_action = agent.get_action(state, True)
                next_state, reward, done, info = env.step(clipped_action)

                states.append(state)
                actions.append(action)
                temp_rewards.append(reward)

                state = next_state
                score += reward

            score_logger.write([step, score])
            scores.append(score)
            temp_targets = np.zeros_like(temp_rewards)
            ret = 0
            for t in reversed(range(len(temp_rewards))):
                ret = temp_rewards[t] + agent.discount_factor*ret
                temp_targets[t] = ret
            targets += list(temp_targets)

        trajs = [states, actions, targets]
        v_loss, p_objective, kl = agent.train(trajs)

        v_loss_logger.write([ep_step, v_loss])
        p_loss_logger.write([ep_step, p_objective])
        p_losses.append(p_objective)
        v_losses.append(v_loss)
        entropies.append(kl)

        #print(v_loss, p_objective, kl)
        print(np.mean(scores), np.mean(p_losses), np.mean(v_losses), np.mean(entropies))
        graph.update(np.mean(scores), np.mean(p_losses), np.mean(v_losses), np.mean(entropies))
        if (epoch+1)%save_freq == 0:
            agent.save()
            v_loss_logger.save()
            p_loss_logger.save()
            score_logger.save()

    graph.update(0,0,0,0,finished=True)

def test():
    global env_name, save_name, agent_args
    env = gym.make(env_name)
    env.unwrapped.initialize(is_render=True)
    agent = Agent(env, agent_args)

    episodes = int(1e6)

    for episode in range(episodes):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action, clipped_action = agent.get_action(state, False)
            #action, clipped_action = agent.get_action(state, True)
            state, reward, done, info = env.step(clipped_action)
            score += reward
            env.render()
        print("score :",score)

if len(sys.argv)== 2 and sys.argv[1] == 'test':
    test()
else:
    train()
