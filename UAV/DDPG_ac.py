# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:00:35 2020

@author: Admin
"""

import gym
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from AirCombat import AirCombatEnv


class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)
        
    def forward(self, s):
        x = F.sigmoid(self.linear1(s))
        x = F.sigmoid(self.linear2(x))
        x = F.sigmoid(self.linear3(x))
        x = torch.tanh(self.linear4(x))

        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = F.sigmoid(self.linear1(x))
        x = F.sigmoid(self.linear2(x))
        x = F.sigmoid(self.linear3(x))
        x = self.linear4(x)

        return x


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        s_dim = self.env.observation_space.shape[0]
        a_dim = self.env.action_space.shape[0]

        self.actor = Actor(s_dim, 256, a_dim)
        self.actor_target = Actor(s_dim, 256, a_dim)
        self.critic = Critic(s_dim+a_dim, 256, a_dim)
        self.critic_target = Critic(s_dim+a_dim, 256, a_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr = self.critic_lr)
        self.buffer = []
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
    def act(self, s0):
        s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)
        a0 = self.actor(s0).squeeze(0).detach().numpy()
        return a0
    
    def put(self, *transition): 
        if len(self.buffer)== self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)
    
    def learn(self):
        if len(self.buffer) < self.batch_size:
            return 
        
        samples = random.sample(self.buffer, self.batch_size)
        
        s0, a0, r1, s1 = zip(*samples)
        
        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size,-1)
        s1 = torch.tensor(s1, dtype=torch.float)
        
        def critic_learn():
            a1 = self.actor_target(s1).detach()
            y_true = r1 + self.gamma * self.critic_target(s1, a1).detach()
            
            y_pred = self.critic(s0, a0)
            
            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()
            
        def actor_learn():
            loss = -torch.mean( self.critic(s0, self.actor(s0)) )
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()
                                           
        def soft_update(net_target, net, tau):
            for target_param, param  in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)
                                           

env = AirCombatEnv()
env.reset()


params = {
    'env': env,
    'gamma': 0.99, 
    'actor_lr': 0.1, 
    'critic_lr': 0.1,
    'tau': 0.8,
    'capacity': 10000, 
    'batch_size': 112,
    }

agent = Agent(**params)

reward_=[]
episode_=[]


Episode = 500
Step = 100

for episode in range(Episode):
    
    s0 = env.reset()
    episode_reward = 0
    
    #fig = plt.figure()
    #ax1 = plt.axes(projection='3d')
    X_a = []
    Y_a = []
    Z_a = []
    X_b = []
    Y_b = []
    Z_b = []
    
    for step in range(Step):

        a0 = agent.act(s0)
        s1, r1, done, _ = env.step(a0)

        agent.put(s0, a0, r1, s1)

        episode_reward += r1 
        s0 = s1
        
        #print(s0)
        #print(a0)
    
        
        agent.learn()
    
        X_a.append(s0[0])
        Y_a.append(s0[1])
        Z_a.append(s0[2])
        X_b.append(s0[6])
        Y_b.append(s0[7])
        Z_b.append(s0[8])
        
    if episode == Episode-1:
        fig = plt.figure()
        ax1 = plt.axes(projection='3d')
        ax1.plot3D(X_a, Y_a, Z_a, 'red')
        ax1.plot3D(X_b, Y_b, Z_b, 'blue')

        plt.show()

    reward_.append(episode_reward)
    episode_.append(episode)

    print(episode, ': ', episode_reward)
plt.figure(2)    
plt.plot(episode_,reward_)
plt.xlabel('episode')
plt.ylabel('reward')