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

env_name = 'HalfCheetah-v2'

save_name = env_name.split('-')[0]
agent_args = {'agent_name':'PPO',
            'env_name':save_name,
            'discount_factor':0.99,
            'hidden1':64,
            'hidden2':64,
            'v_lr':1e-3,
            'p_lr':1e-4,
            'value_epochs':80,
            'policy_epochs':10,
            'clip_value':0.2,
            'gae_coeff':0.97,
            'ent_coeff':0.0,
            }

def train():
    global env_name, save_name, agent_args
    env = gym.make(env_name)
    agent = Agent(env, agent_args)

    p_loss_logger = Logger(save_name, 'p_loss')
    v_loss_logger = Logger(save_name, 'v_loss')
    kl_logger = Logger(save_name, 'kl')
    score_logger = Logger(save_name, 'score')
    graph = Graph(1000, save_name, ['score', 'policy loss', 'value loss', 'kl divergence', 'entropy'])
    episodes = 10
    max_steps = 4000
    max_ep_len = min(1000, env.spec.max_episode_steps)
    epochs = int(1e5)
    save_freq = 10

    save_period = 10
    p_losses = deque(maxlen=save_period)
    v_losses = deque(maxlen=save_period)
    kl_divergence = deque(maxlen=save_period)
    entropies = deque(maxlen=save_period)
    scores = deque(maxlen=save_period*episodes)

    for epoch in range(epochs):
        states = []
        actions = []
        targets = []
        next_states = []
        rewards = []
        gaes = []
        ep_step = 0
        #for episode in range(episodes):
        while ep_step < max_steps:
            state = env.reset()
            done = False
            score = 0
            step = 0
            temp_rewards = []
            values = []
            while True:
                step += 1
                ep_step += 1
                action, clipped_action, value = agent.get_action(state, True)
                next_state, reward, done, info = env.step(clipped_action)

                states.append(state)
                actions.append(action)
                temp_rewards.append(reward)
                next_states.append(next_state)
                rewards.append(reward)
                values.append(value)

                state = next_state
                score += reward

                if done or step >= max_ep_len:
                    break

            if step >= max_ep_len:
                action, clipped_action, value = agent.get_action(state, True)
            else: #중간에 끝난 거면, 다 돌기전에 죽어버린거니, value = 0 으로 해야함
                value = 0
                print("done before max_ep_len...") 
            next_values = values[1:] + [value]
            temp_gaes, temp_targets = agent.get_gaes_targets(temp_rewards, values, next_values)
            targets += list(temp_targets)
            gaes += list(temp_gaes)

            score_logger.write([step, score])
            scores.append(score)

        trajs = [states, actions, targets, next_states, rewards, gaes]
        p_loss, v_loss, kl, entropy = agent.train(trajs)

        p_loss_logger.write([ep_step, p_loss])
        v_loss_logger.write([ep_step, v_loss])
        kl_logger.write([ep_step, kl])
        p_losses.append(p_loss)
        v_losses.append(v_loss)
        kl_divergence.append(kl)
        entropies.append(entropy)

        print(np.mean(scores), np.mean(p_losses), np.mean(v_losses), np.mean(kl_divergence), np.mean(entropies))
        graph.update([np.mean(scores), np.mean(p_losses), np.mean(v_losses), np.mean(kl_divergence), np.mean(entropies)])
        if (epoch+1)%save_freq == 0:
            agent.save()
            p_loss_logger.save()
            v_loss_logger.save()
            kl_logger.save()
            score_logger.save()

    graph.update(None, finished=True)

def test():
    global env_name, save_name, agent_args
    env = gym.make(env_name)
    agent = Agent(env, agent_args)

    episodes = int(1e6)

    for episode in range(episodes):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action, clipped_action, value = agent.get_action(state, False)
            #action, clipped_action, value = agent.get_action(state, True)
            state, reward, done, info = env.step(clipped_action)
            score += reward
            env.render()
            time.sleep(0.01)
        print("score :",score)

if len(sys.argv)== 2 and sys.argv[1] == 'test':
    test()
else:
    train()
