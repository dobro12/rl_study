# to ignore warning message
import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
###########################

from graph_drawer import Graph
from logger import Logger
from nets_ver2 import Agent

from collections import deque
import numpy as np
import pickle
import time
import sys
import gym

#env_name = 'HalfCheetah-v2'
env_name = 'HalfCheetah-v3'

save_name = '{}_ver2'.format(env_name.split('-')[0])
agent_args = {'agent_name':'SAC',
            'env_name':save_name,
            'discount_factor':0.99,
            'hidden1':256,
            'hidden2':256,
            'q_lr':1e-3,
            'p_lr':1e-3,
            'alpha':0.05, #0.2,
            'soft_update':0.005,
            'batch_size':256, #100,
            'replay_memory':1e5,
            }

def train():
    global env_name, save_name, agent_args
    env = gym.make(env_name)
    agent = Agent(env, agent_args)

    score_logger = Logger(save_name, 'score')
    graph = Graph(1000, save_name, ['score', 'policy loss', 'Q value loss', 'entropy'])
    max_steps = 4000
    max_ep_len = min(1000, env.spec.max_episode_steps)
    start_training_after_steps = 1000
    step_per_training = 50
    epochs = 1000
    save_freq = 1

    record_length = 10
    p_losses = deque(maxlen=record_length*int(max_ep_len/step_per_training))
    q_losses = deque(maxlen=record_length*int(max_ep_len/step_per_training))
    entropies = deque(maxlen=record_length*int(max_ep_len/step_per_training))
    scores = deque(maxlen=record_length)

    total_step = 0 
    for epoch in range(epochs):
        ep_step = 0
        while ep_step < max_steps:
            state = env.reset()
            score = 0
            step = 0
            while True:
                step += 1
                ep_step += 1
                total_step += 1
                action = agent.get_action(state, True)
                next_state, reward, done, info = env.step(action)
                done = False if step >= max_ep_len else done

                agent.replay_memory.append([state, action, reward, np.float(done), next_state])

                if len(agent.replay_memory) > start_training_after_steps and (total_step + 1)%step_per_training == 0:
                    for _ in range(step_per_training):
                        p_loss, q_loss, entropy = agent.train()
                    p_losses.append(p_loss)
                    q_losses.append(q_loss)
                    entropies.append(entropy)
                    print(np.mean(scores), np.mean(p_losses), np.mean(q_losses), np.mean(entropies))

                state = next_state
                score += reward

                if done or step >= max_ep_len:
                    break

            score_logger.write([step, score])
            scores.append(score)

            graph.update([np.mean(scores), np.mean(p_losses), np.mean(q_losses), np.mean(entropies)])

        if (epoch+1)%save_freq == 0:
            agent.save()
            score_logger.save()

    graph.update(None, finished=True)

def test():
    global env_name, save_name, agent_args
    env = gym.make(env_name)
    agent = Agent(env, agent_args)

    episodes = int(1e2)

    for episode in range(episodes):
        state = env.reset()
        done = False
        score = 0
        step = 0
        while step < 1000:
            step += 1
            action = agent.get_action(state, False)
            #action, clipped_action, value = agent.get_action(state, True)
            state, reward, done, info = env.step(action)
            score += reward
            env.render()
            time.sleep(0.01)
        print("score :",score)

if len(sys.argv)== 2 and sys.argv[1] == 'test':
    test()
else:
    train()
