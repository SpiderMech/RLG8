# implement basic DQN
from collections import namedtuple, deque
from itertools import count
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# for now we may leave this here, later we might want to move it as we implement different versions of memory
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, h, w, outputs, gamma, eps_start, eps_end, eps_decay, target_update, device):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.device = device

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        # here it simply computes the input size of 
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    # feed forward operation: take input -> produce output
    # tensor = multi-dimensional array. 1st order tensor = vector, 2nd order tensor = matrices
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x))) # relu is used as activation function
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
    

    def setup_model(self):
        # here we have to set up behavioural and target net
        # memory
        pass

    
    def learn(self):
        pass
    

    def save(self):
        pass
