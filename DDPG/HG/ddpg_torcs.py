from gym_torcs import TorcsEnv

import argparse
import numpy as np
import os
import pandas as pd
import random


import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from matplotlib import pyplot as plt
from utils import soft_update, OrnsteinUhlenbeckProcess



class ActorNet(nn.Module):
    def __init__(self, args):
        super(ActorNet, self).__init__()
        state_dim = args.state_dim
        action_dim = args.action_dim
        num_hidden1 = args.hidden1
        num_hidden2 = args.hidden2
        num_hidden3 = args.hidden3
        
        self.actor_layer = nn.Sequential(
            nn.Linear(state_dim, num_hidden1),
            nn.ReLU(),
            nn.Linear(num_hidden1, num_hidden2),
            nn.ReLU(),
            nn.Linear(num_hidden2, num_hidden3),
            nn.ReLU(),
            nn.Linear(num_hidden3, action_dim)
        )
    
    def forward(self, x):
        return self.actor_layer(x)
    
    
class CriticNet(nn.Module):
    def __init__(self, args):
        super(CriticNet, self).__init__()
        state_dim = args.state_dim
        action_dim = args.action_dim
        num_hidden1 = args.hidden1
        num_hidden2 = args.hidden2
        num_hidden3 = args.hidden3
        
        self.critic_layer = nn.Sequential(
            nn.Linear(state_dim+action_dim, num_hidden1),
            nn.ReLU(),
            nn.Linear(num_hidden1, num_hidden2),
            nn.ReLU(),
            nn.Linear(num_hidden2, num_hidden3),
            nn.ReLU(),
            nn.Linear(num_hidden3, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.critic_layer(x)

def sample_minibatch(R, args):
    batch_size = args.batch_size
    minibatch = np.array(random.sample(R, batch_size))

    def array_to_tensor(minibatch, idx):
        if args.cuda:
            return torch.tensor(np.stack(minibatch[:, idx]).astype(np.float32)).cuda()
        else:
            return torch.tensor(np.stack(minibatch[:, idx]).astype(np.float32))

    states = array_to_tensor(minibatch, 0)
    actions = array_to_tensor(minibatch, 1)
    rewards = array_to_tensor(minibatch, 2)
    next_states = array_to_tensor(minibatch, 3)
    dones = array_to_tensor(minibatch, 4)
    return states, actions, rewards, next_states, dones

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--state_dim', default=29, type=int)
    parser.add_argument('--action_dim', default=2, type=int)
    parser.add_argument('--hidden1', default=128, type=int)
    parser.add_argument('--hidden2', default=128, type=int)
    parser.add_argument('--hidden3', default=128, type=int)
    parser.add_argument('--actor_lr', default=1e-4, type=float)
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--tau', default=1e-3, type=float)
    parser.add_argument('--num_episodes', default=5000, type=int)
    parser.add_argument('--max_buff_size', default=100000, type=int)
    parser.add_argument('--warm_up', default=500, type=int)
    parser.add_argument('--decay_rate', default=0.99, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--print_freq', default=1, type=int)
    parser.add_argument('--cuda', default=1, type=int)
    parser.add_argument('--theta', default=0.15, type=float)
    parser.add_argument('--mu', default=0., type=float)
    parser.add_argument('--sigma', default=0.2, type=float)

    args = parser.parse_args()

    '''
    state_dim = args.state_dim
    action_dim = args.action_dim
    hidden1 = args.hidden1
    hidden2 = args.hidden2
    hidden3 = args.hidden3
    actor_lr = args.actor_lr
    critic_lr = args.critic_lr
    batch_size = args.batch_size
    tau = args.tau
    num_episodes = args.num_episodes
    max_buff_size = args.max_buff_size
    warm_up = args.warm_up
    decay_rate = args.decay_rate
    gamma = args.gamma
    print_freq = args.print_freq
    '''
    vision = False

    random_process = OrnsteinUhlenbeckProcess(args)
    criterion = nn.MSELoss()

    cnet = CriticNet(args)
    anet = ActorNet(args)
    target_cnet = CriticNet(args)
    target_anet = ActorNet(args)

    if args.cuda:
        cnet.cuda()
        anet.cuda()
        target_cnet.cuda()
        target_anet.cuda()
        criterion.cuda()

    target_cnet.load_state_dict(cnet.state_dict())
    target_anet.load_state_dict(anet.state_dict())

    critic_optim = optim.Adam(cnet.parameters(), lr=args.critic_lr, weight_decay=1e-5)
    actor_optim = optim.Adam(anet.parameters(), lr=args.actor_lr, weight_decay=1e-5)
    critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=critic_optim, gamma=args.decay_rate)
    actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=actor_optim, gamma=args.decay_rate)

    # training log
    fignum = len([f for f in os.listdir() if 'DDPG_halfcheetah' in f and 'png' in f])
    log_actor_loss = []
    log_critic_loss = []
    log_total_loss = []
    log_ep_return = []
    log_avg_return = []
    log_df = pd.DataFrame(columns=['running', 'EP', 'Loss', 'Return', 'LR'])

    R = []
    ridx = 0
    env = TorcsEnv(vision=vision, throttle=True, gear_change=False)

    for ne in range(args.num_episodes):
        # init random process N
        random_process.reset_states()

        if ne%3==0:
            ob = env.reset(relaunch=True)
        else:
            ob = env.reset()

        obs = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
        state = torch.tensor(obs.astype(np.float32))
        if args.cuda:
            state = state.cuda()

        done = False
        ep_c_loss = []
        ep_a_loss = []
        ep_return = 0.
        
        time_step = 0
        while not done:
            time_step += 1
            state = torch.tensor(obs.astype(np.float32))
            if args.cuda:
                state = state.cuda()
                a = anet(state).cpu().detach().numpy()
            else:
                a = anet(state).detach().numpy()
            noise = random_process.sample()
            action = a + noise

            pre_obs = obs
            ob, reward, done, _ = env.step(action)
            obs = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY,  ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
            ep_return += reward
            
            # [s_t, a_t, r_t, s_(t+1)]
            replay_data = [pre_obs, action, reward, obs, done]
            if len(R)<args.max_buff_size:
                R.append(replay_data)
                if len(R)<args.warm_up:
                    continue
            else:
                R.popleft()
                R.append(replay_data)
            
            states, actions, rewards, next_states, dones = sample_minibatch(R, args)
            
            next_actions = target_anet(next_states) # mu'(s_(i+1))
            next_q_values = target_cnet(next_states, next_actions) # q'(s_(i+1), mu'(s_(i+1)))
            y_targets = rewards + args.gamma * (1-dones) * next_q_values.squeeze()
            q_values = cnet(states, actions).squeeze() # q(s_i, a_i)

            critic_loss = criterion(q_values, y_targets)
            cnet.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            actor_loss = - cnet(states, anet(states)).mean()
            anet.zero_grad()
            actor_loss.backward()
            actor_optim.step()
            
            if args.cuda:
                ep_c_loss.append(critic_loss.cpu().detach().numpy())
                ep_a_loss.append(actor_loss.cpu().detach().numpy())
            else:
                ep_c_loss.append(critic_loss.detach().numpy())
                ep_a_loss.append(actor_loss.detach().numpy())

            # update target_anet, target_cnet
            soft_update(target_cnet, cnet, tau=tau)
            soft_update(target_anet, anet, tau=tau)
            
        ep_c_loss = np.mean(ep_c_loss)
        ep_a_loss = np.mean(ep_a_loss)
        ep_loss = ep_c_loss + ep_a_loss
        
        log_actor_loss.append(ep_a_loss)
        log_critic_loss.append(ep_c_loss)
        log_total_loss.append(ep_loss)
        log_ep_return.append(ep_return)
        log_avg_return.append(np.mean(log_ep_return[-10:]))
        
        
        if (ne+1)%args.print_freq==0:
            print('%d/%d episodes. (%.2f%%)'%(ne+1, args.num_episodes, (ne+1)/args.num_episodes*100))
            #print('Current learning rate:', optimizer.param_groups[0]['lr'])
            print('Total loss:\t', ep_loss)
            print('Critic\t\tActor')
            print('%.2f\t\t%.2f'%(ep_c_loss, ep_a_loss))
            print('Epside Return: [%.1f]'%ep_return)
            print('Episode Length: %d'%time_step)
            print('Lap Time:', env.lapTime)
            print('Num Lap:', env.num_lap)
            print()
            
            f, axes = plt.subplots(4,1)
            f.set_figheight(15)
            f.set_figwidth(10)
            
            axes[0].plot(log_ep_return, color='pink')
            axes[0].plot(log_avg_return, color='red')
            axes[1].plot(log_total_loss, color='blue')
            axes[2].plot(log_critic_loss, color='orange')
            axes[3].plot(log_actor_loss, color='green')

            axes[0].set_title('Episode Return')
            axes[1].set_title('Total Loss')
            axes[2].set_title('Critic Loss')
            axes[3].set_title('Actor Loss')
            plt.savefig('DDPG_torch_%d.png'%fignum)
            
            
            raw_data = ['%.1f%%'%((ne+1)/args.num_episodes*100), int(ne+1), ep_loss, ep_return, actor_optim.param_groups[0]['lr']]
            log_df = log_df.append(pd.Series(raw_data, index = log_df.columns), ignore_index=True)
        critic_scheduler.step()            
        actor_scheduler.step()
