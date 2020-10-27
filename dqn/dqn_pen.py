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
    
    def __init__(self, num_states, num_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_states, 16)
        self.fc2 = nn.Linear(16,64)
        self.fc3 = nn.Linear(64, 256) #中間層
        self.fc4 = nn.Linear(256,256)
        self.fc5 = nn.Linear(256, num_actions)

        #activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h1 = self.relu(self.fc1(x)) # 活性化関数にはReLu
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        h4 = self.relu(self.fc4(h3))
        output = self.fc5(h4)
        return output

import random 
import torch 
from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import namedtuple
from copy import deepcopy

device2 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cpu")
# device = torch.device("cpu")
print(device1)
print(device2)

Transition = namedtuple('Transition',('state','action','state_next', 'reward'))


BACTH_SIZE = 32
CAPACITY = 10000
GAMMA = 0.9
MAX_STEP = 400
NUM_EPISODES = 1000

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


class Brain:
    def __init__(self, num_states, num_actions):
        self.memory = ReplayMemory(CAPACITY)

        self.num_actions = num_actions

        self.main_q_net = Net(num_states,num_actions).to(device1)
        self.tgt_q_net = deepcopy(self.main_q_net).to(device2)
        self.tgt_q_net.load_state_dict(self.main_q_net.state_dict())
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
            action = torch.tensor([[random.randrange(self.num_actions)]],device=device1,dtype=torch.long)

        return action
    

    def replay(self):
        if len(self.memory) < BACTH_SIZE:
            return
        
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_state = self.make_minibatch()

        self.expected_state_action_values = self.get_expected_state_action_values()

        self.update_main_q_network()

    def make_minibatch(self):

        transitions = self.memory.sample(BACTH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_state =torch.cat([s for s in batch.state_next if s is not None]).to(device2)

        return batch, state_batch, action_batch, reward_batch, non_final_next_state
    
    def get_expected_state_action_values(self):

        self.main_q_net.eval()
        self.state_action_values = self.main_q_net(self.state_batch).gather(1, self.action_batch)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, self.batch.state_next)),dtype=torch.uint8)
        next_state_values = torch.zeros(BACTH_SIZE,device=device2)
        # a_m = torch.zeros(BACTH_SIZE).type(torch.LongTensor)

        # a_m = self.main_q_net(self.state_batch).detach().max(1)[1].view(-1,1)
        self.tgt_q_net.eval()
        next_state_values[non_final_mask] = self.tgt_q_net(self.non_final_next_state).max(1)[0].detach()

        expected_state_action_values = self.reward_batch + GAMMA * next_state_values
        # tmp = expected_state_action_values.to(device2)
        expected_state_action_values = torch.tensor(expected_state_action_values, device=device1).detach()

        return expected_state_action_values
    
    def update_main_q_network(self):

        self.main_q_net.train()

        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def update_tgt_q_network(self):

        self.tgt_q_net.load_state_dict(self.main_q_net.state_dict())

class Agent:
    def __init__(self,num_states,num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_q_functions(self):
        self.brain.replay()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state,episode)

        return action
    
    def memorize(self, state, action, state_next, reward):
        self.brain.memory.push(state, action, state_next, reward)
    
    def update_tgt_q_functions(self):
        self.brain.update_tgt_q_network()


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
        # print(reward)
        return reward
    
    def get_state(self):
        return [[self.th, self.th_]]

    def update_state(self, action):

        power = 0.005* np.sign(action)

        self.th_ += -self.g*np.sin(self.th)+power
        self.th_old = self.th
        self.th += self.th_

    def ani(self):
        """
        アニメーション用に現在の状況をSVGで返す
        """
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        fig = plt.figure(figsize=(6, 6))
        th = self.th.numpy().tolist()
        c = (3,3)
        x = c[0]+1.5*np.sin(th[0])
        y = c[1]+1.5*np.cos(th[0])
        plt.plot([c[0],x],[c[1],y])
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
        state = self.env.get_state()
        state = torch.tensor(state,device=device1,dtype=torch.float)
        self.episode = episode
        total_reward = 0
        self.actions=[-1,1] 

        for _ in range(MAX_STEP):

            action = self.agent.get_action(state,self.episode)
            actions = self.actions[action] 
            self.env.update_state(actions)
            reward = self.env.get_reward()
            total_reward += reward
            reward = torch.tensor([reward],device=device2,dtype=torch.float)


            state_next = self.env.get_state()
            state_next = torch.FloatTensor(state_next).to(device1)
            # state_next = torch.tensor(state_next,device=device1,dtype=torch.float)


            self.agent.memorize(state, action, state_next, reward)

            if train:
                self.agent.update_q_functions()

            # if movie:
            #     self.env.ani()
            #     time.sleep(0.01)

            state = state_next

        if train and self.episode % 10 == 0:
            self.agent.update_tgt_q_functions()

        return total_reward

    def run_gif(self, episode, train=True, movie=False):
        self.env.reset(0,0)
        state = self.env.get_state()
        state = torch.tensor(state,dtype=torch.float)
        self.episode = episode
        total_reward = 0
        self.log=[]
        self.actions=[-1,1] 

        for _ in range(MAX_STEP):

            action = self.agent.get_action(state,self.episode)
            actions = self.actions[action] 
            self.env.update_state(actions)
            reward = self.env.get_reward()
            total_reward += reward
            reward = torch.tensor([reward],dtype=torch.float)


            state_next = self.env.get_state()
            state_next = torch.tensor(state_next,dtype=torch.float)


            self.agent.memorize(state, action, state_next, reward)
            s = state[0].numpy()
            a = action[0].numpy()
            r = reward.numpy()
            print(state,action,reward)
            print(s,a,r)
            self.log.append(np.hstack([s[0], a, r]))

            if train:
                self.agent.update_q_functions()

            # if movie:
            #     self.env.ani()
            #     time.sleep(0.01)

            state = state_next

        if train and self.episode % 10 == 0:
            self.agent.update_tgt_q_functions()

        return total_reward,self.log

# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# fig = plt.figure(figsize=(6, 6))
# th = self.th.numpy().tolist()
# c = (3,3)
# x = c[0]+1.5*np.sin(th[0])
# y = c[1]+1.5*np.cos(th[0])
# plt.plot([c[0],x],[c[1],y])
# plt.show()

import time
if __name__ == '__main__':
    num_states = 2
    num_actions = 2
    t1 = time.time()
    agent = Agent(num_states, num_actions)
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
            t2 = time.time()
            if test_highscore<total_reward:
                print("highscore!")
                torch.save(agent.brain.main_q_net, "model/%06d_hs.model" %episode)
                # serializers.save_npz('model/%06d_hs.model'%episode, agent.model)
                test_highscore=total_reward
            # print(episode)
            # print(total_reward)
            print('Finished %d Episode: Total reward = %.2f: Elasped time = %.2f' %(episode, total_reward, t2-t1))
            t1 = time.time()

            out=("%d,%d\n" % (episode,total_reward))
            fw.write(out)
            fw.flush()
    fw.close






