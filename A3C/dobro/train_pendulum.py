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
import threading
import pickle
import time
import sys
import gym

env_name = 'Pendulum-v0'
agent_args = {'hidden1':64,
            'hidden2':64,
            'v_lr':1e-3,
            'p_lr':1e-4,
            'init_std':0.0}

def train():
    global total_step, total_max_step, env_name, global_agent, step_period, gamma, \
            loss_logger, score_logger, graph, p_losses, v_losses, entropies, scores, agent_args
    gamma = 0.9
    num_thread = 10
    total_step = 0
    total_max_step = 1e8
    step_period = 1e2 #1e4
    step_period = int(step_period / num_thread)
    save_name = env_name.split('-')[0]

    env = gym.make(env_name)
    global_agent = Agent("global", env, save_name, gamma, agent_args)
    loss_logger = Logger(save_name, 'loss')
    score_logger = Logger(save_name, 'score')
    graph = Graph(1000, save_name.upper(), 'A3C')
    env.close()

    save_period = 1000
    p_losses = deque(maxlen=save_period)
    v_losses = deque(maxlen=save_period)
    entropies = deque(maxlen=save_period)
    scores = deque(maxlen=save_period)

    def thread_func(t_idx):
        global total_step, total_max_step, env_name, global_agent, step_period, gamma, \
                loss_logger, score_logger, graph, p_losses, v_losses, entropies, scores, agent_args
        env = gym.make(env_name)
        agent = Agent("local_{}".format(t_idx), env, save_name, gamma, agent_args)
        episode = 0
        step = 0

        p_loss = None
        v_loss = None
        entropy = None

        #gradient reset & parameter synchronize
        agent.update_parameter(global_agent)
        start_step = step
        states = []
        actions = []
        rewards = []
        dones =[]
        next_states = []

        score = 0
        state = env.reset()
        while total_step < total_max_step:
            step += 1
            total_step += 1

            action = agent.get_action(state, True)
            next_state, reward, done, info = env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state) 
            score += reward

            if step-start_step == step_period:
                ret = 0 if done else agent.get_value(next_state)
                targets = []
                state_values = []
                for i in range(len(states)):
                    if dones[-i-1]:
                        #ret = 0
                        ret = agent.get_value(next_states[-i-1])
                        state_values.append(ret)
                    ret = rewards[-i-1] + gamma*ret
                    targets.append(ret)
                targets = targets[::-1]
                p_grad, p_loss, v_grad, v_loss, entropy = agent.calc_gradient(states, actions, targets)
                p_losses.append(p_loss)
                v_losses.append(v_loss)
                entropies.append(entropy)
                global_agent.update_with_gradients(p_grad, v_grad)
                #loss_logger.write([step-start_step,p_loss,v_loss])
                agent.update_parameter(global_agent)
                if t_idx == 0:
                    graph.update(np.mean(scores), np.mean(p_losses), [np.mean(v_losses), np.mean(state_values)], np.mean(entropies))

                start_step = step
                states = []
                actions = []
                rewards = []
                dones = []

            state = next_state
            #score_logger.write([cnt, score])
            if done:
                episode += 1
                if t_idx == 0 and episode%10 == 0: 
                    global_agent.save()
                scores.append(score)
                print(t_idx,score)
                score = 0
                state = env.reset()

    threads = []
    for i in range(num_thread):
        threads.append(threading.Thread(target=thread_func, args=(i,)))
        threads[-1].start()

    for thread in threads:
        thread.join()
    graph.update(0,0,0,0,True)


def test():
    global env_name, agent_args
    save_name = env_name.split('-')[0]
    gamma = 0.99
    env = gym.make(env_name)
    agent = Agent("global", env, save_name, gamma, agent_args)
    episodes = int(1e6)

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            #action = agent.get_action(state, False)
            action = agent.get_action(state, True)
            print(action)
            #time.sleep(0.01)
            #if action[0] > 0:
            #    a_t = 1
            #else :
            #    a_t = 0
            state, reward, done, info = env.step(action)
            #state, reward, done, info = env.step(a_t)
            env.render()
            #time.sleep(1)

if len(sys.argv)== 2 and sys.argv[1] == 'test':
    test()
else:
    train()
