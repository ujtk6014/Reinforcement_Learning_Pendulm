import numpy as np
import gym 
import matplotlib.pyplot as plt


import torch.nn as nn
import torch.nn.functional as F

STATE_NUM = 4

class Net(nn.Module):
    def __init__(self, n_in = STATE_NUM):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(n_in, 16)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 1024)
        self.fc4 = nn.Linear(1024, 2)
    
    def forward(self,x):
        h1 = F.leaky_relu(self.fc1(x))
        h2 = F.leaky_relu(self.fc2(h1))
        h3 = F.leaky_relu(self.fc3(h2))
        h4 = F.leaky_relu(self.fc4(h3))
        output = self.fc4(h4)

        return output

import random 
import torch 
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import namedtuple

Transisiton = namedtuple('Transition',('state','action','state_next', 'reward'))


BACTH_SIZE = 32
CAPACITY = 10000

class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):

        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[index] = Transition(state,action,state_next, reward)
        self.index = (self.index + 1) % self.capacity
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

TD_ERROR_EPSILON = 0.0001

class TDerrorMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, td_error):

        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[index] = td_error
        self.index = (self.index + 1) % self.capacity
    
    def __len__(self):
        return len(self.memory)
    
    def get_prioritized_indexes(self, batch_size):

        sum_absolute_td_error = np.sum(np.absolute(self.memory))
        sum_absolute_td_error += TD_ERROR_EPSILON * len(self.memory)

        rand_list = np.random.uniform(0, sum_absolute_td_error, batch_size)
        rand_list = np.sort(rand_list)

        indexes = []
        idx = 0

        tmp_sum_absolute_td_error = 0
        for rand_num in rand_list:
            while tmp_sum_absolute_td_error < rand_num:
                tmp_sum_absolute_td_error += abs(self.memory[idx]) + TD_ERROR_EPSILON)
                idx += 1

                if idx >= len(self.memory):
                    idx = len(self.memory) -1
                indexes.append(idx)
        return indexes
    
    def update_td_error(self, updated_td_errors):
        self.memory = updated_td_errors
                



class Brain:
    def __init__(self):
        self.memory = ReplayMemory(CAPACITY)
        self.td_error_memory = TDerrorMemory(CAPACITY)

        self.main_q_net = Net()
        self.tgt_q_net = Net()
        print(self.main_q_net)
        print(self.tgt_q_net)
        self.optimizer = optim.Adam(self.main_q_net.parameters())
    
    def decide_action(self, state, episode):
        epsilon = 0.5 * (1/episode + 1)

        if epsilon <= np.random.uniform(0,1):
            self.main_q_net.eval()
            with torch.no_grad():
                action = self.main_q_net(state).max(1)[1]

    def replay(self, episode):
        if len(self.experienceMemory) < BACTH_SIZE:
            return
        
        self.batch, self.state_batch, self.action_batch, self.reward_batch,
        = self.make_minibatch(episode)

        self.expected_state_action_values = self.get_expected_state_action_values()

        self.update_main_q_network()

    def make_minibatch(self, epidode):

        if episode < 30:
            transitions = self.memory.sample(BACTH_SIZE)

        else:
            indexes = self.td_error_memory.get_prioritized_indexes(BACTH_SIZE)
            transitions[self.memory.memory[n] for n in indexes]
        
        batch = Transtion(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        return batch, state_batch, action_batch, reward_batch
     
        

        



