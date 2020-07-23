import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class DQN_Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=100, fc2_units=100):
        super(DQN_Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        # self.ln1 = nn.LayerNorm(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        # self.ln2 = nn.LayerNorm(fc2_units)
        self.fc3 = nn.Linear(fc1_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        x = state
        x = self.fc1(x)
        # x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = self.ln2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=25, fc2_units=100):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        # self.ln1 = nn.LayerNorm(fc1_units)
        # self.fc2 = nn.Linear(fc1_units, fc2_units)
        # self.ln2 = nn.LayerNorm(fc2_units)
        self.fc3 = nn.Linear(fc1_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        # self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = state
        x = self.fc1(x)
        # x = self.ln1(x)
        x = F.relu(x)
        # x = self.fc2(x)
        # # x = self.ln2(x)
        # x = F.relu(x)
        x = self.fc3(x)
        return torch.tanh(x)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=100, fc2_units=100):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            fc3_units (int): Number of nodes in the third hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.fc1(state)
        xs = self.bn1(xs)
        xs = F.leaky_relu(xs)
        x = torch.cat((xs, action), dim=1)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        return self.fc3(x)

if __name__ == '__main__':
    from thop import profile
    net = Critic(2, 1, 0).to('cuda')

    input_size = 2
    input = torch.randn(1, input_size).to('cuda')
    action = torch.rand(1, 1).to('cuda')
    flops, params = profile(net, inputs=(input, action, ))
# #print('flops: {}, params: {}'.format(flops, params))
    print('flops: {}, params: {}'.format(flops, params))