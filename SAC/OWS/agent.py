import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sac_network import network_basic
from sac_network import policy_network
import visdom
import time

class Agent:
    def __init__(self, **kwargs):
        self.state_dim = kwargs['state_dim']
        self.action_dim = kwargs['action_dim']
        self.action_range = kwargs['action_range']
        self.learning_rate = kwargs['learning_rate']
        self.gamma = kwargs['gamma']
        self.set_seed(kwargs['seed'])
        self.set_network()

    def set_seed(self,seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def set_network(self):
        pass
    def decide_action(self):
        pass
    def cal_loss(self):
        pass
    def update_param(self):
        pass
    def set_train(self):
        pass
    def set_eval(self):
        pass
    def train(self):
        pass
    def save_agent(self):
        pass
    def load_agent(self):
        pass

class sac_Agent(Agent):

    def __init__(self,**kwargs):
        super(sac_Agent,self).__init__(**kwargs)
        self.action_mean = torch.Tensor([num.mean() for num in self.action_range])
        self.action_scale = torch.Tensor([(num[0]-num[1])/2. for num in self.action_range])
        self.target_update = kwargs['target_update']
        self.target_update_interval =  kwargs['target_update_interval']
        self.gradients_step =  kwargs['gradients_step']
        self.path = kwargs['path']
        self.alpha = kwargs['alpha']
        self.value_loss_sum = 0
        self.policy_loss_sum = 0
        self.q1_loss_sum = 0
        self.q2_loss_sum = 0
        self.loss_cnt = 0
        self.value_loss_avg = 0
        self.q1_loss_avg = 0
        self.q2_loss_avg = 0
        self.policy_loss_avg = 0
        if(torch.cuda.is_available()):
            self.action_mean = self.action_mean.cuda()
            self.action_scale= self.action_scale.cuda()

    def set_network(self):
        print("network setting")
        self.value_network = network_basic(self.state_dim, 1)
        self.target_value_network = network_basic(self.state_dim, 1)
        self.q_network1 = network_basic(self.state_dim+self.action_dim, 1)
        self.q_network2 = network_basic(self.state_dim+self.action_dim, 1)
        self.policy = policy_network(self.state_dim , self.action_dim)

        self.value_network.initialize()
        self.q_network1.initialize()
        self.q_network2.initialize()
        self.policy.initialize()

        self.target_value_network.load_state_dict(self.value_network.state_dict())
        
        if(torch.cuda.is_available()):
            self.value_network = self.value_network.cuda()
            self.target_value_network = self.target_value_network.cuda()
            self.q_network1 = self.q_network1.cuda()
            self.q_network2 = self.q_network2.cuda()
            self.policy = self.policy.cuda()

        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.learning_rate)
        self.q1_optimizer = optim.Adam(self.q_network1.parameters(), lr=self.learning_rate)
        self.q2_optimizer = optim.Adam(self.q_network2.parameters(), lr=self.learning_rate)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    def decide_action(self,state):
        if torch.is_tensor(state):
            state = torch.Tensor(state).cuda() if torch.cuda.is_available() and not state.is_cuda else state
        else:
            state = torch.Tensor(state).cuda() if torch.cuda.is_available() else torch.Tensor(state)
        num_batch = 1 if len(state.shape)==1 else state.shape[0]
        mu,var = self.policy(state)
        mu = mu.squeeze()
        var = var.squeeze()
        noise = torch.Tensor([ [ torch.normal(mean=torch.zeros(self.action_dim)).numpy() ] for i in range(num_batch) ] )
        noise = noise.squeeze()
        if(len(mu.shape) == 0):
            mu = mu.unsqueeze(-1)
            var = var.unsqueeze(-1)
            noise = noise.unsqueeze(-1)
        if(torch.cuda.is_available()):
            noise=noise.cuda()
        unbounded_action = var**(1./2)  * noise + mu        
        bounded_action = torch.tanh(unbounded_action) * self.action_scale + self.action_mean  
        return bounded_action, unbounded_action, mu, var 

    def cal_loss(self, batch):
        
        state, action, ub_action, next_state, reward, done, mu_buf, var_buf = self.decode_batch(batch)

        value = self.value_network(state)

        q_value1_from_buffer = self.q_network1(torch.cat([state,action],-1))
        q_value2_from_buffer = self.q_network2(torch.cat([state,action],-1))
        log_policy_value_from_buffer = self.alpha * self.get_log_policy_value(action,ub_action,mu_buf,var_buf)

        value_loss = (0.5*(value - (q_value1_from_buffer.detach() + q_value2_from_buffer.detach())/2 - log_policy_value_from_buffer.detach())**2).mean()

        target_value_next_state = self.target_value_network(next_state) 
        target_q = reward + self.gamma * target_value_next_state

        q1_loss = (0.5 * (q_value1_from_buffer - target_q.detach())**2).mean()
        q2_loss = (0.5 * (q_value2_from_buffer - target_q.detach())**2).mean()


        bounded_action,unbounded_action, mu, var = self.decide_action(state)

        log_policy_value_from_noise = self.alpha * self.get_log_policy_value(bounded_action, unbounded_action, mu, var)
        q_min =torch.min( self.q_network1(torch.cat([state,bounded_action],-1)) ,  self.q_network2(torch.cat([state,bounded_action],-1)) )

        policy_loss = (log_policy_value_from_noise - q_min).mean()
        self.value_loss_sum += value_loss.detach().cpu()
        self.q1_loss_sum += q1_loss.detach().cpu()
        self.q2_loss_sum += q2_loss.detach().cpu()
        self.policy_loss_sum += policy_loss.detach().cpu()
        self.loss_cnt += 1

        return value_loss, q1_loss, q2_loss, policy_loss
 
   
    def train(self, batch, epoch, t, done):
        self.set_train()

        self.update_param(batch) if t % self.gradients_step is 0 else True

        self.target_net_update() if t % self.target_update_interval is 0 else True

        if(done):
            #self.learning_rate = self.learning_rate * self.gamma
            self.save_agent( self.path, epoch)
            if(self.loss_cnt != 0):
                print("value_loss : {0:.4f} , q1_loss : {1:.4f} , q2_loss : {2:.4f} , policy_loss : {3:.4f} ".format\
                        (self.value_loss_sum.item()/self.loss_cnt, self.q1_loss_sum.item()/self.loss_cnt , self.q2_loss_sum.item()/self.loss_cnt, self.policy_loss_sum.item()/self.loss_cnt))
                self.value_loss_avg , self.q1_loss_avg, self.q2_loss_avg, self.policy_loss_avg = (self.value_loss_sum.item()/self.loss_cnt, self.q1_loss_sum.item()/self.loss_cnt , self.q2_loss_sum.item()/self.loss_cnt, self.policy_loss_sum.item()/self.loss_cnt)
            self.initialize_loss_sum()

    def get_plt_loss_values(self):
        return self.value_loss_avg , self.q1_loss_avg, self.q2_loss_avg, self.policy_loss_avg

    def initialize_loss_sum(self):
        self.value_loss_sum = 0
        self.q1_loss_sum = 0
        self.q2_loss_sum = 0
        self.policy_loss_sum = 0
        self.loss_cnt = 0

    def decode_batch(self, batch):
        size = len(batch)
        state = torch.empty([size,self.state_dim])
        action =  torch.empty([size,self.action_dim])
        ub_action =  torch.empty([size,self.action_dim])
        next_state = torch.empty([size,self.state_dim])
        reward =  torch.empty([size,1])
        done =  torch.empty([size,1])
        mu_buf =  torch.empty([size,self.action_dim])
        var_buf =  torch.empty([size,self.action_dim])

        for i in range(size):
            state[i], action[i] , ub_action[i] ,next_state[i] , reward[i] ,  done[i] , mu_buf[i], var_buf[i] = batch[i]
        state = state.cuda() if torch.cuda.is_available() else state
        action =action.cuda() if torch.cuda.is_available() else action
        ub_action =ub_action.cuda() if torch.cuda.is_available() else ub_action
        next_state =next_state.cuda() if torch.cuda.is_available() else next_state
        reward =reward.cuda() if torch.cuda.is_available() else reward
        done =done.cuda() if torch.cuda.is_available() else done
        mu_buf =mu_buf.cuda() if torch.cuda.is_available() else mu_buf
        var_buf =var_buf.cuda() if torch.cuda.is_available() else var_buf
        return state, action, ub_action, next_state, reward, done, mu_buf, var_buf

    def update_param(self,batch):
        value_loss, q1_loss, q2_loss, policy_loss = self.cal_loss(batch)
        value_loss.backward()

        q1_loss.backward()
        q2_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        policy_loss.backward()
        self.value_optimizer.step()
        self.policy_optimizer.step()

    def set_train(self):
        self.q_network1.train()
        self.q_network2.train()
        self.value_network.train()
        self.policy.train()
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        self.policy_optimizer.zero_grad()

    def set_eval(self):
        self.q_network1.eval()
        self.q_network2.eval()
        self.value_network.eval()
        self.policy.eval()
    
    def get_log_policy_value(self, action, unbounded_action , mu, var ):
        ones = torch.ones_like(unbounded_action).cuda() if torch.cuda.is_available() else torch.ones_like(unbounded_action)
        jacobian_sum = torch.log(torch.clamp( ((ones - torch.tanh(unbounded_action)**2)* self.action_scale.unsqueeze(0).expand_as(ones) ), 1e-10) ).sum(1).unsqueeze(1)
        log_prob_unbounded = -0.5*(unbounded_action - mu)**2/var - torch.log(torch.clamp(2*3.141592*var , 1e-10))
        log_prob_action = log_prob_unbounded - jacobian_sum.expand_as(unbounded_action)
        return log_prob_action
        
    def target_net_update(self):    
        for t_v_params , l_v_params in zip(self.target_value_network.parameters() , self.value_network.parameters()):
            t_v_params.data.copy_( t_v_params.data * (1-self.target_update) + self.target_update * l_v_params.data )
                
    def save_agent(self,path,epoch):
        torch.save({
            'epoch' : epoch,
            'value_network_state_dict':self.value_network.state_dict(),
            'target_value_network_state_dict':self.target_value_network.state_dict(),
            'q_network1_state_dict':self.q_network1.state_dict(),
            'q_network2_state_dict':self.q_network2.state_dict(),
            'policy_state_dict':self.policy.state_dict()}, path)

    def load_agent(self,path):
        checkpoint = torch.load(path)
        epoch = checkpoint['epoch']
        self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
        self.target_value_network.load_state_dict(checkpoint['target_value_network_state_dict']),
        self.q_network1.load_state_dict(checkpoint['q_network1_state_dict']),
        self.q_network2.load_state_dict(checkpoint['q_network2__state_dict']),
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        return epoch

        



