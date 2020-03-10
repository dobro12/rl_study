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
import sys
import gym

env_name = 'Pendulum-v0'
#env_name = 'DobroHalfCheetah-v0'
save_name = env_name.split('-')[0]
agent_args = {'agent_name':'DDPG',
            'env_name':save_name,
            'discount_factor':0.99,
            'hidden1':64,
            'hidden2':64,
            'v_lr':1e-3,
            'p_lr':1e-4,
            'soft_update':0.01}

def train():
    global env_name, save_name, agent_args
    env = gym.make(env_name)
    if env_name == 'DobroHalfCheetah-v0':
        env.unwrapped.initialize(is_render=False)
    agent = Agent(env, agent_args)

    v_loss_logger = Logger(save_name, 'v_loss')
    p_loss_logger = Logger(save_name, 'p_loss')
    score_logger = Logger(save_name, 'score')
    graph = Graph(1000, save_name.upper(), agent.name)
    episodes = int(5e2)
    save_freq = 10

    save_period = 100
    p_losses = deque(maxlen=save_period)
    v_losses = deque(maxlen=save_period)
    entropies = deque(maxlen=save_period)
    scores = deque(maxlen=save_period)

    for episode in range(episodes):
        state = env.reset()
        done = False
        score = 0
        step = 0

        while not done:
            step += 1
            action = agent.get_action(state, True)
            next_state, reward, done, info = env.step(action)
            agent.replay_memory.append([np.array(state, np.float32), action, reward, done, np.array(next_state, np.float32)])
            ########################

            if len(agent.replay_memory) > agent.train_start:
                v_loss, p_loss = agent.train()
                v_loss_logger.write([1, v_loss])
                p_loss_logger.write([1, p_loss])
                p_losses.append(p_loss)
                v_losses.append(v_loss)
                value = agent.get_value(state, action)
                entropies.append(value)
                scores.append(reward)
                graph.update(np.mean(scores), np.mean(p_losses), np.mean(v_losses), np.mean(entropies))
            state = next_state
            score += reward

        print(episode, score, agent.epsilon)
        score_logger.write([step, score])
        if (episode+1)%save_freq == 0:
            agent.save()
            v_loss_logger.save()
            p_loss_logger.save()
            score_logger.save()

    graph.update(0,0,0,0,finished=True)

def test():
    global env_name, save_name, agent_args
    env = gym.make(env_name)
    if env_name == 'DobroHalfCheetah-v0':
        env.unwrapped.initialize(is_render=True)
    agent = Agent(env, agent_args)
    agent.epsilon = 0.01

    episodes = int(1e6)
    avg_Q = deque(maxlen=200)

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.get_action(state, False)
            state, reward, done, info = env.step(action)
            env.render()

if len(sys.argv)== 2 and sys.argv[1] == 'test':
    test()
else:
    train()