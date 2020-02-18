import gym
import time
from a2c import Agent
import sys

if len(sys.argv) != 2:
    raise ValueError('test or train?')

    
env = gym.make('CartPole-v1')
agent = Agent(env)

if sys.argv[1] == 'train':
    agent.train()
else:
    agent.test()
