from collections import deque
from logger import Logger
from nets import Agent
import numpy as np
import pickle
import sys
import gym

#env_name = 'MountainCarContinuous-v0'
env_name = 'Pendulum-v0'
env = gym.make(env_name)

def train():
    global env, env_name
    env_name = env_name.split('-')[0]
    agent = Agent(env, env_name)
    loss_logger = Logger(env_name, 'loss')
    score_logger = Logger(env_name, 'score')
    action_low = env.action_space.low[0]
    action_high = env.action_space.high[0]
    episodes = int(5e2)
    avg_Q = deque(maxlen=200)

    for episode in range(episodes):
        state = env.reset()
        done = False
        score = 0
        step = 0

        while not done:
            step += 1
            action = agent.get_action(state)
            a_t = (action/(agent.n_action-1))
            a_t = a_t*(action_high - action_low) + action_low
            next_state, reward, done, info = env.step([a_t])

            agent.replay_memory.append([np.array(state, np.float32), action, reward, done, np.array(next_state, np.float32)])
            ########################

            #replay 메모리에 어느정도 쌓이면 학습시작하기
            if len(agent.replay_memory) > agent.train_start:
                Q, loss = agent.train()
                loss_logger.write([1, loss])
                avg_Q.append(Q)
            state = next_state
            score += reward

        #print(episode, accumulate+100, self.epsilon)
        print(episode, score, agent.epsilon, np.mean(avg_Q))
        agent.update_target_model()
        score_logger.write([step, score])
        if (episode+1)%agent.save_freq == 0:
            agent.save()
            loss_logger.save()
            score_logger.save()

def test():
    agent = Agent(env)
    agent.epsilon = 0.01
    action_low = env.action_space.low[0]
    action_high = env.action_space.high[0]
    episodes = int(1e6)
    avg_Q = deque(maxlen=200)

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.get_action(state)
            a_t = (action/(agent.n_action-1))
            a_t = a_t*(action_high - action_low) + action_low
            state, reward, done, info = env.step([a_t])
            env.render()

if len(sys.argv)== 2 and sys.argv[1] == 'test':
    test()
else:
    train()