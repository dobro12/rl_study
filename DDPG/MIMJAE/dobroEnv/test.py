import subprocess
import sys
GITPATH = subprocess.run('git rev-parse --show-toplevel'.split(' '), \
        stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n','')
sys.path.append(GITPATH)
import dobroEnv
##############################################################

import numpy as np
import threading
import time
import gym

'''
# render 하지 않는 경우
env = gym.make('DobroHalfCheetah-v0')
env.unwrapped.initialize(is_render=False)

state = env.reset()
while True:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    if done: 
        break
print("Finished one episode.")
env.close()
########################
'''

# render 하는 경우
env = gym.make('DobroHalfCheetah-v0')
env.unwrapped.initialize(is_render=True)

state = env.reset()
while True:
    #action = env.action_space.sample()
    action = np.zeros(6)
    state, reward, done, info = env.step(action)
    if done: 
        break
print("Finished one episode.")
env.close()
########################

'''
# multithread 일 경우
def func(idx):
    env = gym.make('DobroHalfCheetah-v0')
    env.unwrapped.initialize(is_render=False)
    for i in range(3):
        score = 0
        state = env.reset()
        while True:
            action = env.action_space.sample()
            #action = np.zeros(6)
            state, reward, done, info = env.step(action)
            score += reward
            if done: 
                break
        print(idx, score)
    env.close()

num_thread = 3
threads = []
for i in range(num_thread):
    threads.append(threading.Thread(target=func, args=(i+1,)))
    threads[-1].start()

for thread in threads:
    thread.join()
'''
