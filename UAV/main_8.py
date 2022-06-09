# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 16:58:16 2021

@author: 86153
"""

import torch
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import numpy as np
import random
import math
import gym

from AirCombat import AirCombatEnv
from AC_env import AirCombatEnv

from SAC_8 import config,SAC
import  datetime
from torch.utils.tensorboard import SummaryWriter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
writer1 = SummaryWriter(
    log_dir='runs/{}_SAC_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'UCAV',
                                        'SAC_step_reward'))
writer2 = SummaryWriter(
    log_dir='runs/{}_SAC_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'UCAV',
                                       'SAC_episode_reward'))

def run(env, agent):
    ep=0
    Return = []
    target = 5
    step = 0
    E = []
    action_mean = (env.action_space.high - env.action_space.low) / 2
    action_bias = (env.action_space.high + env.action_space.low) / 2
    state_mean = (env.state_space.high - env.state_space.low) / 2
    state_bias = (env.state_space.high + env.state_space.low) / 2
    
    for episode in range(config['max_episode']):
        ep+=1
        E.append(episode)
        
        # X_a=[]
        # Y_a=[]
        # Z_a=[]
        # X_b=[]
        # Y_b=[]
        # Z_b=[]
        
        score = 0
        state = env.reset(episode)
        # action_mean = (env.action_space.high - env.action_space.low) / 2
        # action_bias = (env.action_space.high + env.action_space.low) / 2
        for i in range(config['max_step']):
            state_norm = (state - state_bias)/state_mean
            action = agent.get_action(state_norm)
            #print(action)
            
            next_state, reward, done, _ = env.step(action * action_mean + action_bias, episode)
            # print(next_state)
            # print(reward)
            
            #next_state, reward, done, _ = env.step(action) 

            done_mask = 0 if done else 1
            
            state_norm = (state - state_bias)/state_mean
            next_state_norm = (next_state - state_bias)/state_mean

            
            agent.push((state_norm, action, reward, next_state_norm, done_mask))
            
            
            # state_= np.array([(state[0]+1000)/2000, (state[1]+1000)/2000, (state[2]+1000)/2000, state[3]/(2*np.pi), (state[4]+np.pi/4)/(np.pi/2), (state[5]-10)/70, (state[6]+1000)/2000, (state[7]+1000)/2000, (state[8]+1000)/2000, state[9]/(2*np.pi), (state[10]+np.pi/4)/(np.pi/2), (state[11]-10)/70, state[12]/1000])             
            # agent.push((state_, action, reward, next_state, done_mask))   #这里将归一化后的state输入buffer
            # print(state_)    
            # print(state) 
            state = next_state



            

       
            
            score += reward
            step +=1
            if done:
                break
            if agent.buffer.buffer_len() > 2000:
                agent.update()
        
        


        # #绘制动态轨迹图
        # if episode+1 == config['max_episode']:
        #     fig=plt.figure(figsize=(10,10))
        #     ax1 = Axes3D(fig)
        #     ax1.plot3D([0,1],[0,1],[0,1], 'red')
        #     j=0
        #     i=0

        #     a_track = np.array([[X_a[0],Y_a[0],Z_a[0]]])
        #     a_track_s = np.array([.0,.0,.0])

        #     b_track = np.array([[X_b[0],Y_b[0],Z_b[0]]])
        #     b_track_s = np.array([.0,.0,.0])


        #     def gen_path():
        #         global a_track_s,a_track,j,b_track_s,b_track
        #         j+=1
            
         
        #         x_1 = 0
        #         y_1 = 0
        #         z_1 = 0
        #         x_2 = 0
        #         y_2 = 0
        #         z_2 = 0

                
            
        #         x_1 = X_a[j]
        #         y_1 = Y_a[j]
        #         z_1 = Z_a[j]
            
        #         x_2 = X_b[j]
        #         y_2 = Y_b[j]
        #         z_2 = Z_b[j] 

                


            
        #         a_track_s =[x_1,y_1,z_1]
        
        #         a_track = np.append(a_track, [a_track_s],axis=0)
            

                
        #         b_track_s =[x_2,y_2,z_2]
        
        #         b_track = np.append(b_track, [b_track_s],axis=0)    
            
        #         return a_track, b_track

        #     def update(i):
            
        #     # label = 'timestep {0}'.format(i)
        #     #print("step:",i)
        #         i+=1
        #     # 更新直线和x轴（用一个新的x轴的标签）。
        #     # 用元组（Tuple）的形式返回在这一帧要被重新绘图的物体
        #         a_track,b_track = gen_path()
        #     # ax1.set_xlabel(label)
        
        #         ax1.plot3D(a_track[:, 0], a_track[:, 1], a_track[:, 2], 'blue')
        #         ax1.plot3D(b_track[:, 0], b_track[:, 1], b_track[:, 2], 'red')
            
            
        #         ax1.set_xlabel("X Axis")
        #         ax1.set_ylabel("Y Axis")
        #         ax1.set_zlabel("Z Axis")
        #         ax1.set_title("Trajectory")
        #         ax1.set(xlim=[round(min(min(X_a),min(X_b)))-1,math.ceil(max(max(X_a),max(X_b)))], 
        #                 ylim=[round(min(min(Y_a),min(Y_b)))-1,math.ceil(max(max(Y_a),max(Y_b)))], 
        #                 zlim=[round(min(min(Z_a),min(Z_b)))-1,math.ceil(max(max(Z_a),max(Z_b)))])      
            
            
        #         return ax1

                
        #     anim = FuncAnimation(fig, update, frames=np.arange(0,len(X_a)), interval=0.1, repeat = False)
        # # anim = FuncAnimation(fig, update, frames=len(X_a), interval=0.1, repeat=False)
            
        #     anim.save(r'C:\Users\86153\Desktop\AirCombat_test\GIF\b.gif')  
        # # plt.show()



        if (episode+1)%2000==0:
            agent.save_nets('./',episode)        
                       
        if episode+1==config['max_episode']:
            agent.save_nets('./',episode)

        Return.append(score)
        print("episode:{}, Return:{}, Buffer_len:{}".format(episode, score, agent.buffer.buffer_len()))

        writer1.add_scalar('train/episode_reward',score, step)
        writer2.add_scalar('train_2/episode_reward',score, ep)
    print("=======================end=======================")
    plt.figure(2)
    plt.plot(E,Return)
    plt.legend()
    plt.show()


    # env and RL param
torch.set_num_threads(config['num_threads'])
os.environ['MKL_NUM_THREADS'] = str(config['num_threads'])
if not config['random_seed']:
    torch.cuda.manual_seed_all(config['seed'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
# method
env = AirCombatEnv()
sac = SAC(env)
run(env,sac)