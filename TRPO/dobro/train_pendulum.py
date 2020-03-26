# to ignore warning message
import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
###########################

from graph_drawer import Graph
from logger import Logger
from nets import Agent

from collections import deque
import numpy as np
import pickle
import time
import sys
import gym

env_name = 'Pendulum-v0'

save_name = env_name.split('-')[0]
agent_args = {'agent_name':'TRPO',
            'env_name':save_name,
            'discount_factor':0.9,
            'hidden1':1,
            'hidden2':1,
            'v_lr':1e-3,
            'batch_size':128,
            'std':1.0}

def train():
    global env_name, save_name, agent_args
    env = gym.make(env_name)
    agent = Agent(env, agent_args)

    v_loss_logger = Logger(save_name, 'v_loss')
    p_loss_logger = Logger(save_name, 'p_loss')
    score_logger = Logger(save_name, 'score')
    graph = Graph(1000, save_name.upper(), agent.name)
    episodes = 1
    epochs = int(1e3)
    save_freq = 10

    save_period = 1000
    p_losses = deque(maxlen=save_period)
    v_losses = deque(maxlen=save_period)
    entropies = deque(maxlen=save_period)
    scores = deque(maxlen=save_period)

    for epoch in range(epochs):
        states = []
        actions = []
        rewards = []
        next_states = []
        ep_step = 0
        for episode in range(episodes):
            state = env.reset()
            done = False
            score = 0
            step = 0
            while not done:
                step += 1
                ep_step += 1
                action = agent.get_action(state, True)
                next_state, reward, done, info = env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)

                state = next_state
                score += reward

            score_logger.write([step, score])
            scores.append(score)

        trajs = [states, actions, rewards, next_states]
        v_loss, p_objective, kl = agent.train(trajs)

        v_loss_logger.write([ep_step, v_loss])
        p_loss_logger.write([ep_step, p_objective])
        p_losses.append(p_objective)
        v_losses.append(v_loss)
        entropies.append(kl)

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
    if env_name == 'DobroHalfCheetah-v0':
        env.unwrapped.initialize(is_render=True)
    elif env_name == 'HalfCheetahBulletEnv-v0':
        env.render()
    agent = Agent(env, agent_args)

    episodes = int(1e6)
    avg_Q = deque(maxlen=200)

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            #action = agent.get_action(state, False)
            action = agent.get_action(state, True)
            state, reward, done, info = env.step(action)
            print(np.mean(action))
            env.render()
            time.sleep(0.01)

if len(sys.argv)== 2 and sys.argv[1] == 'test':
    test()
else:
    train()
