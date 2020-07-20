import numpy as np
import torch
import pickle
import gym
from agent import sac_Agent
import argparse
import os
import visdom

parser = argparse.ArgumentParser(description='training')
parser.add_argument('-seed', type=int, help='seed')
parser.add_argument('-path', help='file path') 
args = parser.parse_args()

path = os.path.join(os.getcwd() , args.path)

def get_env_param(env):
    state_dim_tuple = env.observation_space.shape
    state_dim = state_dim_tuple[0] if len(state_dim_tuple) == 1 else state_dim_tuple
    action_high = torch.Tensor(env.action_space.high).unsqueeze(1)
    action_low = torch.Tensor(env.action_space.low).unsqueeze(1)
    action_range = torch.cat([action_high, action_low],1)
    action_dim = env.action_space.shape[0]
    return state_dim, action_dim, action_range

env=gym.make('Pendulum-v0')
state_dim, action_dim, action_range = get_env_param(env)


seed = 0
epoch = 150
episode = 1000
buffer_size = 5000
batch_size = 128
agent_setting_parameters = { 
        'state_dim' : state_dim,
        'action_dim' : action_dim,
        'action_range' : action_range,
        'learning_rate' : 0.001,
        'gamma' : 0.99,
        'seed' : seed,
        'target_update' : 0.005,
        'target_update_interval' : 1,
        'gradients_step' : 1,
        'path' : path,
        'alpha' : 0.1
        }
print(agent_setting_parameters)
def update_buffer(element, replay_buffer, index, full):
    if(full):
        replay_buffer[index] = element
    else:
        replay_buffer.append(element)
vis = visdom.Visdom()

value_loss_plt = None 
q1_loss_plt = None
q2_loss_plt = None
policy_loss_plt = None
total_reward_plt = None

def visualize_loss(seed, epoch, value_loss, q1_loss, q2_loss, policy_loss, reward_sum):
    global value_loss_plt
    global q1_loss_plt
    global q2_loss_plt
    global policy_loss_plt
    global total_reward_plt
    if(epoch==0):
        value_loss_plt = vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([value_loss]) ,env = 'sac'+str(seed) ,opts=dict(title="value_loss"))
        q1_loss_plt = vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([q1_loss]) ,env = 'sac'+str(seed), opts=dict(title="q1_loss"))
        q2_loss_plt = vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([q2_loss]) ,env = 'sac'+str(seed), opts=dict(title="q2_loss"))
        policy_loss_plt = vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([policy_loss]) ,env = 'sac'+str(seed), opts=dict(title="policy_loss"))
        total_reward_plt = vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([reward_sum]),env = 'sac'+str(seed), opts = dict(title="total_reward"))
        
    else:
        vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([value_loss]) , win=value_loss_plt,env = 'sac'+str(seed), update = "append")
        vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([q1_loss]) , win=q1_loss_plt,env = 'sac'+str(seed), update = "append")
        vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([q2_loss]), win=q2_loss_plt,env = 'sac'+str(seed), update = "append")
        vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([policy_loss]) ,win=policy_loss_plt,env = 'sac'+str(seed), update = "append")
        vis.line(X=torch.Tensor([epoch]), Y=torch.Tensor([reward_sum]), win=total_reward_plt,env = 'sac'+str(seed), update = "append")



total_reward = [[] for i in range(5)]
print(total_reward)
for seed in range(5):
    agent = sac_Agent(**agent_setting_parameters)
    for i in range(epoch):
        print(i)
        reward_sum = 0
        obs = env.reset()
        replay_buffer = [] 
        buffer_index = 0
        buffer_full = False
        for t in range(episode):
            action, ub_action, mu, var = agent.decide_action(obs)
            if(torch.cuda.is_available()):
                action = action.detach().cpu()
                ub_action = ub_action.detach().cpu()
                mu = mu.detach().cpu()
                var = var.detach().cpu()
            next_obs, reward, done, _ = env.step(action)
            element = (torch.Tensor(obs), torch.Tensor(action), torch.Tensor(ub_action), torch.Tensor(next_obs), torch.Tensor([reward]), torch.Tensor([done]), torch.Tensor(mu), torch.Tensor(var))
            update_buffer(element, replay_buffer, buffer_index, buffer_full)
            buffer_index += 1 
            if(buffer_index is buffer_size):
                buffer_full = True
                buffer_index = 0
            reward_sum += reward
    
            if(batch_size < buffer_index or buffer_full):
                indices = np.random.choice(len(replay_buffer),batch_size, replace=False)
                batch = [replay_buffer[i] for i in indices]
                agent.train(batch,i,t,done)
            if(done):
                break
            obs = next_obs
        value_loss, q1_loss, q2_loss, policy_loss = agent.get_plt_loss_values()
        print(reward_sum)
        visualize_loss(seed,i,value_loss,q1_loss,q2_loss,policy_loss,reward_sum)
        total_reward[seed].append(reward_sum)
    
    env.close()
with open('sac_reward_pendulum.txt', 'wb') as f:
    pickle.dump(total_reward,f)
