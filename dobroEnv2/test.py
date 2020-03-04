import subprocess
import sys
GITPATH = subprocess.run('git rev-parse --show-toplevel'.split(' '), \
        stdout=subprocess.PIPE).stdout.decode('utf-8').replace('\n','')
sys.path.append(GITPATH)
import dobroEnv2
##############################################################

import env
import time
import gym

env = gym.make('DobroQuadruped-v0')
env.unwrapped.initialize(is_render=True)

state = env.reset()
while True:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    #if done: break