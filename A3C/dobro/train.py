from collections import deque
from logger import Logger
from nets import Agent
import numpy as np
import threading
import pickle
import sys
import gym

env_name = 'Pendulum-v0'

def train():
    global total_step, total_max_step, env_name, global_agent, step_period, gamma, \
            loss_logger, score_logger
    gamma = 0.99
    num_thread = 10
    total_step = 0
    total_max_step = 1e6
    step_period = 1e3
    step_period = int(step_period / num_thread)

    env = gym.make(env_name)
    global_agent = Agent(env, env_name, gamma)
    loss_logger = Logger(env_name, 'loss')
    score_logger = Logger(env_name, 'score')
    env.close()

    def thread_func(t_idx):
        global total_step, total_max_step, env_name, global_agent, step_period, gamma, \
                loss_logger, score_logger
        env = gym.make(env_name)
        agent = Agent(env, env_name, gamma)
        step = 0

        while total_step < total_max_step:
            #gradient reset & parameter synchronize
            agent.update_parameter(global_agent)
            ###
            start_step = step
            states = []
            actions = []
            rewards = []
            score = 0
            cnt = 0
            state = env.reset()
            while True:
                cnt += 1
                step += 1
                total_step += 1
                action = agent.get_action(state)
                next_state, reward, done, info = env.step(action)
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                score += reward
                if done or step-start_step == step_period:
                    ret = 0 if done else agent.get_value(next_state)
                    targets = []
                    for i in range(len(states)):
                        ret = rewards[-i-1] + gamma*ret
                        targets.append(ret)
                    targets = targets[::-1]
                    gradients, p_loss, v_loss = agent.calc_gradient(states, actions, targets)
                    global_agent.update_with_gradients(gradients)
                    loss_logger.write([step-start_step,p_loss,v_loss])
                    if done:
                        break
                    agent.update_parameter(global_agent)
                    start_step = step
                    states = []
                    actions = []
                    rewards = []
                state = next_state
            score_logger.write([cnt, score])


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