import subprocess
import sys
GITPATH = subprocess.run('git rev-parse --show-toplevel'.split(' '), \
        stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n','')
sys.path.append(GITPATH)
##############################################################

import dobroEnv
import time
import gym

env = gym.make('DobroHalfCheetah-v0')

state = env.reset()
while True:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    env.render()
    time.sleep(1e-2)

    if done: break