import numpy as np
import matplotlib.pyplot as plt
import svgwrite as sw
from IPython import display
from IPython.display import SVG
import time

import torch.nn as nn
import torch.nn.functional as F

STATE_NUM = 2

class Net(nn.Module):
    def __init__(self, n_in = STATE_NUM):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(n_in, 16)
        self.fc2 = nn.Linear(16,64)
        self.fc3 = nn.Linear(64, 256)
        self.fc4 = nn.Linear(256, 1024)
        self.fc5 = nn.Linear(1024, 2)
    
    def forward(self,x):
        h1 = F.leaky_relu(self.fc1(x))
        h2 = F.leaky_relu(self.fc2(h1))
        h3 = F.leaky_relu(self.fc3(h2))
        h4 = F.leaky_relu(self.fc4(h3))
        output = F.leaky_relu(self.fc5(h4))

        return output

import random 
import torch 
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import namedtuple

Transition = namedtuple('Transition',('state','action','state_next', 'reward'))


BACTH_SIZE = 32
CAPACITY = 10000
GAMMA = 0.9
MAX_STEP = 300
NUM_EPISODES = 500

class ReplayMemory:
    def __init__(self, CAPACITY):
        self.capacity = CAPACITY
        self.memory = []
        self.index = 0

    def push(self, state, action, state_next, reward):

        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.index] = Transition(state,action,state_next, reward)
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

        self.memory[self.index] = td_error
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
                tmp_sum_absolute_td_error += (abs(self.memory[idx]) + TD_ERROR_EPSILON)
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

        self.num_actions = 2

        self.main_q_net = Net()
        self.tgt_q_net = Net()
        print(self.main_q_net)
        print(self.tgt_q_net)
        self.optimizer = optim.Adam(self.main_q_net.parameters())
    def decide_action(self, state, episode):
        epsilon = 0.5 *( 1/(episode + 1) )

        if epsilon <= np.random.uniform(0,1):
            self.main_q_net.eval()
            with torch.no_grad():
                # print(state)
                # b = self.main_q_net(state)
                # print(b)
                action = self.main_q_net(state).max(1)[1].view(1,1)

        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]])

        return action
    

    def replay(self, episode):
        if len(self.memory) < BACTH_SIZE:
            return
        
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.next_state_batch = self.make_minibatch(episode)

        self.expected_state_action_values = self.get_expected_state_action_values()

        self.update_main_q_network()

    def make_minibatch(self, epidode):

        if episode < 30:
            transitions = self.memory.sample(BACTH_SIZE)

        else:
            indexes = self.td_error_memory.get_prioritized_indexes(BACTH_SIZE)
            transitions = [self.memory.memory[n] for n in indexes]
        
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch =torch.cat(batch.state_next)

        return batch, state_batch, action_batch, reward_batch, next_state_batch
    
    def get_expected_state_action_values(self):

        self.main_q_net.eval()
        self.tgt_q_net.eval()
        
        self.state_action_values = self.main_q_net(self.state_batch).gather(1, self.action_batch)

        next_state_values = torch.zeros(BACTH_SIZE)
        # a_m = torch.zeros(BACTH_SIZE).type(torch.LongTensor)

        # a_m = self.main_q_net(self.state_batch).detach().max(1)[1].view(-1,1)

        next_state_values = self.tgt_q_net(self.next_state_batch).max(1)[0].detach()

        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        return expected_state_action_values
    
    def update_main_q_network(self):

        self.main_q_net.train()

        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def update_tgt_q_network(self):

        self.tgt_q_net.load_state_dict(self.main_q_net.state_dict())

    def update_td_error_memory(self):
        self.main_q_net.eval()
        self.tgt_q_net.eval()

        transitions = self.memory.memory
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch =torch.cat(batch.state_next)

        state_action_values = self.main_q_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(len(self.memory))
        # a_m = torch.zeros(BACTH_SIZE).type(torch.LongTensor)

        # a_m = self.main_q_net(self.state_batch).detach().max(1)[1].view(-1,1)

        next_state_values = self.tgt_q_net(self.next_state_batch).max(1)[0].detach().squeeze()
        # print(state_action_values.size())
        # print(next_state_values.size())
        # print(reward_batch.size())
        td_errors = (reward_batch + GAMMA * next_state_values) - state_action_values.squeeze()

        self.td_error_memory.memory = td_errors.detach().numpy().tolist()


class Agent:
    def __init__(self):
        self.brain = Brain()

    def update_q_functions(self, episode):
        self.brain.replay(episode)

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)

        return action
    
    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)
    
    def update_tgt_q_functions(self):
        self.brain.update_tgt_q_network()

    def memorize_td_error(self, td_error):
        self.brain.td_error_memory.push(td_error)

    def update_td_error_memory(self):
        self.brain.update_td_error_memory()

class pendulumEnvironment:

    def __init__(self):
        self.reset(0,0)

    def reset(self, initial_theta, initial_dtheta):
        self.th = initial_theta
        self.th_old = self.th
        self.th_ = initial_dtheta
        self.g = 0.01
        self.highscore = -1.0

    def get_reward(self):
        reward = 0
        h = -np.cos(self.th)
        if h >= 0:
            reward = 5*np.abs(h)

        else:
            reward = -np.abs(h)
        
        return reward
    
    def get_state(self):
        return [[self.th, self.th_]]

    def update_state(self, action):

        power = 0.005* np.sign(action)

        self.th_ += -self.g*np.sin(self.th)+power
        self.th_old = self.th
        self.th += self.th_
        # print(self.th, self.th_)

    def ani(self):
        """
        アニメーション用に現在の状況をSVGで返す
        """
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        fig = plt.figure()
        anim = animation.FuncAnimation

        plt.show()
        # dr=sw.Drawing("hoge.svg",(150,150))
        # c=(75,75)
        # th = self.th.numpy().tolist()
        # dr.add(dr.line(c,(c[0]+50*np.sin(th[0]),c[1]+50*np.cos(th[0])), stroke=sw.utils.rgb(0,0,0),stroke_width=3))
        # return SVG(dr.tostring())
        
class simulator:
    def __init__(self, environment, agent):
        self.env = environment
        self.agent = agent


    def run(self, episode, train=True, movie=False):
        self.env.reset(0,0)
        state = env.get_state()
        state = torch.FloatTensor(state)
        self.episode = episode
        total_reward = 0

        for _ in range(MAX_STEP):

            action = self.agent.get_action(state, self.episode)

            self.env.update_state(action)
            reward = self.env.get_reward()
            reward += reward
            reward = torch.FloatTensor([reward])


            state_next = env.get_state()
            state_next = torch.FloatTensor(state_next)


            self.agent.memorize(state, action, state_next, reward)

            self.agent.memorize_td_error(0)

            if train:
                self.agent.update_q_functions(self.episode)

            # if movie:
            #     self.env.ani()
            #     time.sleep(0.01)

            state = state_next
            
        self.agent.update_td_error_memory()

        if train and self.episode % 2 == 0:
            self.agent.update_tgt_q_functions()

        return total_reward
        
if __name__ == '__main__':
    agent = Agent()
    env = pendulumEnvironment()
    sim = simulator(env,agent)

    test_highscore=0

    fw=open("log.csv","w")

    for episode in range(NUM_EPISODES):
        total_reward=sim.run(episode, train=True,movie=True)

        # if episode%1000 ==0:
            # serializers.save_npz('model/%06d.model'%episode, agent.model)

        if episode%10 == 0:
            total_reward=sim.run(episode, train=False, movie=False)
            if test_highscore<total_reward:
                print("highscore!")
                
                # serializers.save_npz('model/%06d_hs.model'%episode, agent.model)
                test_highscore=total_reward
            print(episode)
            print(total_reward)

            out=("%d,%d\n" % (episode,total_reward))
            fw.write(out)
            fw.flush()
    fw.close






