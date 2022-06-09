import os
import argparse
import collections
import pickle
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from copy import deepcopy
from gym import spaces
from scipy.io import loadmat, savemat

config = {
    # For RL
    'device': 'cuda:0',  # ['cpu','cuda:0','cuda:2',....]
    'num_threads': 1,  # if use cpu
    'buffer_maxlen': 1000000,
    'batch_size': 512,
    'mid_dim': 256,
    'alpha': 1,  # alpha<=1 (only for sac temperature param)
    'tau': 0.01,  # target smoothing coefficient
    'q_lr': 0.001,  # q net learning rate
    'a_lr': 0.001,  # actor net learning rate
    'gamma': 0.98,  # discounted factor
    'max_episode': 200000,  # max episode
    'max_step': 100,  # fix 100 day
    'random_seed': False,  # False mean's fix seed [False,True]
    'seed': 1,  # fix seed
    # model RL
    'model_num': 5,  # model numbers
    'model_buffer_len': 1000,
    'model_threshold': 0.1,
    'beta': 0.2,  # [0,1)
    'v_lr': 0.001
}


class Buffer:
    def __init__(self, buffer_maxlen, device):
        self.device = torch.device(device)
        self.buffer = collections.deque(maxlen=buffer_maxlen)

    def push(self, data):
        self.buffer.append(data)

    def sample(self, batch_size):
        state_list = []
        action_list = []
        reward_list = []
        next_state_list = []
        done_list = []

        batch = random.sample(self.buffer, batch_size)
        for experience in batch:
            s, a, r, n_s, d = experience
            state_list.append(s)
            action_list.append(a)
            reward_list.append(r)
            next_state_list.append(n_s)
            done_list.append(d)

        state_list = torch.as_tensor(state_list, dtype=torch.float32, device=self.device)
        action_list = torch.as_tensor(action_list, dtype=torch.float32, device=self.device)
        reward_list = torch.as_tensor(reward_list, dtype=torch.float32, device=self.device)
        next_state_list = torch.as_tensor(next_state_list, dtype=torch.float32, device=self.device)
        done_list = torch.as_tensor(done_list, dtype=torch.float32, device=self.device)

        return state_list, action_list, reward_list.unsqueeze(-1), next_state_list, \
               done_list.unsqueeze(-1)

    def buffer_len(self):
        return len(self.buffer)


class SACActor(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net_combine = nn.Sequential(nn.Linear(state_dim, mid_dim), nn.ReLU())
        self.net_mean = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                      nn.Linear(mid_dim, action_dim))
        self.net_log_std = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                         nn.Linear(mid_dim, action_dim))
        self.sqrt_2pi_log = 0.9189385332046727  # =np.log(np.sqrt(2 * np.pi))

    def forward(self, state):
        x = self.net_state(state)
        return self.net_mean(x).tanh()  # action

    def get_action(self, state):
        x = self.net_combine(state)
        mean = self.net_mean(x)
        std = self.net_log_std(x).clamp(-16, 2).exp()
        return torch.normal(mean, std).tanh()

    def get_action_test(self, state):
        x = self.net_combine(state)
        mean = self.net_mean(x)
        return mean.tanh()

    def get_action_log_prob(self, state):
        x = self.net_combine(state)
        mean = self.net_mean(x)
        log_std = self.net_log_std(x).clamp(-16, 2)
        std = log_std.exp()

        # re-parameterize
        noise = torch.randn_like(mean, requires_grad=True)
        a_tan = (mean + std * noise).tanh()  # action.tanh()

        log_prob = log_std + self.sqrt_2pi_log + noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
        log_prob = log_prob + (-a_tan.pow(2) + 1.000001).log()  # fix log_prob using the derivative of action.tanh()
        return a_tan, log_prob.sum(1, keepdim=True)


class TwinCritic(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super(TwinCritic, self).__init__()
        # shared parameter
        self.net_combine = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU())  # concat(state, action)
        self.net_q1 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, 1))  # q1 value
        self.net_q2 = nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                    nn.Linear(mid_dim, 1))  # q2 value

    def forward(self, state, action):
        # only use q1
        x = self.net_combine(torch.cat((state, action), dim=1))
        return self.net_q1(x)

    def get_q1_q2(self, state, action):
        # return q1, q2 value
        x = self.net_combine(torch.cat((state, action), dim=1))
        return self.net_q1(x), self.net_q2(x)


class Deterministic_Model(nn.Module):
    def __init__(self, state_dim, action_dim, mid_dim):
        super().__init__()
        self.net_combine = nn.Sequential(nn.Linear(state_dim + action_dim, mid_dim), nn.ReLU(),
                                         nn.Linear(mid_dim, mid_dim), nn.ReLU(),
                                         nn.Linear(mid_dim, mid_dim))
        self.state = nn.Sequential(nn.Linear(mid_dim, state_dim))
        self.reward = nn.Sequential(nn.Linear(mid_dim, 1))

    def forward(self, state, action):
        x = self.net_combine(torch.cat([state, action], dim=1))
        state = self.state(x)
        reward = self.reward(x)
        return state, reward


def soft_target_update(target, current, tau=0.05):
    for target_param, param in zip(target.parameters(), current.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

class SAC:
    def __init__(self, env):
        self.env = env
        self.device = config['device']
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.mid_dim = config['mid_dim']

        self.gamma = config['gamma']
        self.tau = config['tau']
        self.q_lr = config['q_lr']
        self.a_lr = config['a_lr']
        self.v_lr = config['v_lr']

        # buffer
        self.batch_size = config['batch_size']
        self.buffer = Buffer(config['buffer_maxlen'], self.device)

        self.target_entropy = np.log(self.action_dim)
        self.alpha_log = torch.tensor((-np.log(self.action_dim),), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter

        self.actor = SACActor(self.state_dim, self.action_dim, self.mid_dim).to(self.device)
        
        # #加载训练好的网络的参数，在此基础上继续训练        
        # self.actor.load_state_dict(torch.load('./Models/49999best_actor.pt'))
       
        self.actor_target = deepcopy(self.actor)
        self.critic = TwinCritic(self.state_dim, self.action_dim, int(self.mid_dim)).to(self.device)
        
        # #加载训练好的网络的参数，在此基础上继续训练        
        # self.critic.load_state_dict(torch.load('./Models/49999best_critic.pt'))        
        
        self.critic_target = deepcopy(self.critic)

        self.criterion = torch.nn.SmoothL1Loss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.q_lr)
        self.alpha_optimizer = torch.optim.Adam((self.alpha_log,), lr=self.v_lr)

    def get_action(self, states):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.actor.get_action(states)
        return actions.detach().cpu().numpy()

    def get_action_test(self, states):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.actor.get_action_test(states)
        return actions.detach().cpu().numpy()

    def push(self, data):
        self.buffer.push(data)

    def update(self):
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        alpha = self.alpha_log.exp().detach()

        with torch.no_grad():
            next_action, next_log_prob = self.actor_target.get_action_log_prob(next_state)
            next_q = torch.min(*self.critic_target.get_q1_q2(next_state, next_action))
            q_target = reward + done * (next_q + next_log_prob * alpha) * self.gamma
        q1, q2 = self.critic.get_q1_q2(state, action)
        critic_loss = self.criterion(q1, q_target) + self.criterion(q2, q_target)

        action_pg, log_prob = self.actor.get_action_log_prob(state)  # policy gradient
        alpha_loss = (self.alpha_log * (log_prob - self.target_entropy).detach()).mean()

        alpha = self.alpha_log.exp().detach()
        with torch.no_grad():
            self.alpha_log[:] = self.alpha_log.clamp(-16, 2)
        actor_loss = -(torch.min(*self.critic_target.get_q1_q2(state, action_pg)) + log_prob * alpha).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.mean().backward()
        self.actor_optimizer.step()

        soft_target_update(self.critic_target, self.critic, self.tau)
        soft_target_update(self.actor_target, self.actor, self.tau)

    def save_nets(self, dir_name,episode):
        if not os.path.exists(dir_name + '/Models'):
            os.mkdir(dir_name + '/Models')
        torch.save(self.critic.state_dict(), dir_name + '/Models/' +str(episode)+ 'best_critic.pt')
        torch.save(self.actor.state_dict(), dir_name + '/Models/' +str(episode)+ 'best_actor.pt')
        print('RL saved successfully')

    def load_nets(self, dir_name,episode):
        self.critic.load_state_dict(torch.load(dir_name + '/Models/' +str(episode)+ 'best_critic.pt'))
        self.actor.load_state_dict(torch.load(dir_name + '/Models/' +str(episode)+ 'best_actor.pt'))
        self.critic_target.load_state_dict(torch.load(dir_name + '/Models/' +str(episode)+ 'best_critic.pt'))
        self.actor_target.load_state_dict(torch.load(dir_name + '/Models/' +str(episode)+ 'best_actor.pt'))
        # print('RL load successfully')


class SACwithModel(SAC):
    def __init__(self, env):
        super().__init__(env)
        self.beta = config['beta']
        self.v_lr = config['v_lr']
        self.max_step = config['max_step']

        # buffer
        self.batch_size = config['batch_size']
        self.buffer = Buffer(config['buffer_maxlen'], self.device)
        self.model_buffer = Buffer(config['model_buffer_len'], self.device)
        self.temp = []
        self.step = 0

        self.target_entropy = np.log(self.action_dim)
        self.alpha_log = torch.tensor((-np.log(self.action_dim),), dtype=torch.float32,
                                      requires_grad=True, device=self.device)  # trainable parameter

        self.model_num = config['model_num']
        self.model_threshold = config['model_threshold']
        self.model = []
        self.model_optimizer = []
        for _ in range(self.model_num):
            model = Deterministic_Model(self.state_dim, self.action_dim, self.mid_dim).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=self.v_lr)
            self.model.append(model)
            self.model_optimizer.append(optimizer)

    def get_action(self, states):
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = self.actor.get_action(states)
        return actions.detach().cpu().numpy()

    def push(self, data):
        self.buffer.push(data)
        self.temp.append(data)
        self.step += 1
        if len(self.temp) > self.max_step - 1:
            pre_sum = 0
            for i in range(len(self.temp) - 1, -1, -1):
                r = self.temp[i][2] + 0.99 * pre_sum
                pre_sum = r
                self.model_buffer.push((self.temp[i][0], self.temp[i][1], r / 100, self.temp[i][3], self.temp[i][4]))
            self.temp = []
            if (self.model_buffer.buffer_len() > 300):
                self.model_update()

    # use model
    def update(self):
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        alpha = self.alpha_log.exp().detach()
        # choice model
        model = random.choice(self.model)

        with torch.no_grad():
            model_next_state, model_reward = model(state, action)
            next_action, next_log_prob = self.actor_target.get_action_log_prob(next_state)
            next_q = torch.min(*self.critic_target.get_q1_q2(next_state, next_action))
            q_target = ((1 - self.beta) * reward + (model_reward) * self.beta + done * (
                    next_q + next_log_prob * alpha) * self.gamma)
        q1, q2 = self.critic.get_q1_q2(state, action)

        critic_loss = self.criterion(q1, q_target) + self.criterion(q2, q_target)

        action_pg, log_prob = self.actor.get_action_log_prob(state)  # policy gradient
        alpha_loss = (self.alpha_log * (log_prob - self.target_entropy).detach()).mean()

        alpha = self.alpha_log.exp().detach()
        with torch.no_grad():
            self.alpha_log[:] = self.alpha_log.clamp(-16, 2)
        actor_loss = -(torch.min(*self.critic_target.get_q1_q2(state, action_pg)) + log_prob * alpha).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.mean().backward()
        self.actor_optimizer.step()

        soft_target_update(self.critic_target, self.critic, self.tau)
        soft_target_update(self.actor_target, self.actor, self.tau)

        # return alpha.item(), obj_critic.item()

    def model_update(self):
        model_loss = 1000
        times = 0
        while model_loss > self.model_num * self.model_threshold or times < 50:
            model_loss = 0
            times += 1
            for model, optimizer in zip(self.model, self.model_optimizer):
                state, action, reward, next_state, done = self.model_buffer.sample(128)
                model_state, model_reward = model(state, action)
                loss2 = torch.nn.functional.mse_loss(model_reward, reward)
                loss = (loss2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model_loss = model_loss + loss.item()


