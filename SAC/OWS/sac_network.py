import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
class network_basic(nn.Module):
    
    def __init__(self, in_dim, out_dim):
        super(network_basic,self).__init__()
        self.linear1 = nn.Linear(in_dim,256)
        self.linear2 = nn.Linear(256,256)
        self.linear3 = nn.Linear(256,out_dim)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        out = self.linear3(x)
        return out
    def initialize(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)

class policy_network(network_basic):
    def __init__(self, in_dim, out_dim):
        super(policy_network,self).__init__(in_dim,out_dim)
        self.linear4 = nn.Linear(256,out_dim)
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        variance = F.softplus(self.linear4(x))
        mean = self.linear3(x)
        return mean, variance
    def initialize(self):
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        torch.nn.init.xavier_uniform_(self.linear4.weight)


    
