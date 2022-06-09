# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 13:15:55 2021

@author: 86153
"""
import pprint
from cvxopt import matrix, solvers
import copy
from matplotlib.patches import Circle
import torch
import os
import numpy as np
import random
import gym
import math

from matplotlib.pyplot import MultipleLocator
from CarCombat import CarCombatEnv
from SAC import config, SAC
import  datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


import warnings
warnings.filterwarnings("ignore")
writer1 = SummaryWriter(
    log_dir='runs/{}_SAC_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'car',
                                       'SAC_step_reward'))
writer2 = SummaryWriter(
    log_dir='runs/{}_SAC_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'car',
                                       'SAC_episode_reward'))
                                  


class MPC_QP_solver():
    def __init__(self):
        return
        
        

    def sol(self,x,y,pusin,v,x_r,y_r,v_r,pusin_r,l_r,l_f,T,a_max,delta_max,a_min,delta_min,N_p):  #T是采样周期
 
        # print([x,y,pusin,v,x_r,y_r,pusin_r])
        # delta_r = np.pi/120
        # delta = (action * action_mean + action_bias)[1]
        # delta = action_MPC[1]
        
        # belta = math.atan((l_r / (l_r + l_f)) * math.tan(delta))
        # belta_r = math.atan((l_r / (l_r + l_f)) * math.tan(delta_r))
        delta_r = 0
       #计算A矩阵  
        A = 0
        A = copy.deepcopy(np.array([[1, 0, -v_r * T * math.sin(pusin_r), T * math.cos(pusin_r)],
                      [0, 1, v_r * T * math.cos(pusin_r), T * math.sin(pusin_r)],
                      [0, 0, 1, T * math.tan(delta_r) / (l_r + l_f)],
                      [0, 0, 0, 1]]))
        
        
        #计算B矩阵
        B = 0
        B = copy.deepcopy(np.array([[0, 0],
                      [0, 0],
                      [0, (v_r *T) / ((l_r + l_f)*math.cos(delta_r)**2) ],
                      [T, 0]]))
        
        #计算E矩阵
        E = 0
        E = copy.deepcopy(np.array([[x - x_r],
                      [y - y_r],
                      [pusin - pusin_r],
                      [v - v_r]]))
        
        #计算系数矩阵Q
        Q = 0
        q=copy.deepcopy([5, 5, 0, 5]) * N_p   #从左至右分别表示每一项状态量误差优化的权重（x轴距离偏差，y轴距离偏差，偏航角pusin偏差，速度v偏差）
        Q = np.diag(copy.deepcopy(q))
        
        
        #计算系数矩阵R
        R = 0
        r = copy.deepcopy([0, 0]) * N_p     #从左至右分别表示每一项动作量优化的权重（加速度a，转角delta）
        R = np.diag(copy.deepcopy(r))
        
        #计算G矩阵
        G = 0
        G=copy.deepcopy(A)
        a=copy.deepcopy(A)
        for i_1 in range(N_p-1):
            a=np.dot(copy.deepcopy(A),copy.deepcopy(a))
            G = np.vstack((copy.deepcopy(G), copy.deepcopy(a)))
        # print(G)
        
        
        #构造函数，计算矩阵M的n次方
        def matrix_n_power(M,n):
            if n == 0:
                return np.eye(np.shape(M)[0])
            elif n == 1:
                return copy.deepcopy(M)
            a=copy.deepcopy(M)
            for i_2 in range(n-1):
                a=np.dot(copy.deepcopy(M),copy.deepcopy(a))
            return a
        
        
        #构造函数，计算H矩阵的每一列构成,n表示第几列
        def H_column(A,B,n,N_p):   
            b=copy.deepcopy(B)
            z=np.zeros((np.shape(B)[0], np.shape(B)[1])) 
            if n == N_p-1:
                b = np.vstack((np.zeros((n*np.shape(B)[0], np.shape(B)[1])) ,copy.deepcopy(b)))
                
            elif n == 0:
                for i_3 in range(N_p-1):
                    b = np.vstack((copy.deepcopy(b), np.dot(matrix_n_power(copy.deepcopy(A),i_3+1),copy.deepcopy(B))))
        
            else:
                for i_4 in range(N_p-(n+1)):
                    b = np.vstack((copy.deepcopy(b), np.dot(matrix_n_power(copy.deepcopy(A),i_4+1),copy.deepcopy(B))))
                z = np.zeros((n*np.shape(B)[0], np.shape(B)[1])) 
                b = np.vstack((copy.deepcopy(z),copy.deepcopy(b)))
            return b
        
        # print(N_p)        
                    
        
        #计算H矩阵
        H = 0
        H = copy.deepcopy(H_column(copy.deepcopy(A),copy.deepcopy(B),0,N_p))
        for i_5 in range(N_p-1):
            H = np.hstack((copy.deepcopy(H), copy.deepcopy(H_column(copy.deepcopy(A),copy.deepcopy(B),i_5+1,N_p))))
            
        
        #计算cvxopt求解器中的矩阵：result = solvers.qp(P,q,G,h)
            
        #计算P矩阵
        P_cvx = 0
        q_cvx = 0
        G_cvx = 0
        h_cvx = 0        
        
        P_cvx = 2 * ((np.dot(np.dot((copy.deepcopy(H)).transpose(),copy.deepcopy(Q)),copy.deepcopy(H))) + copy.deepcopy(R))
        q_cvx = 2 * np.dot(np.dot(np.dot((copy.deepcopy(H)).transpose(), copy.deepcopy(Q).transpose()),copy.deepcopy(G)),copy.deepcopy(E))
        G_cvx = np.vstack((np.eye(2*N_p), -np.eye(2*N_p)))
        
        # print(G_cvx)        

        
        action_max = np.array([[a_max],[delta_max]])
        action_min = np.array([[a_min],[delta_min]])
        # print(action_min)
        h_1 = copy.deepcopy(action_max) 
        for i_6 in range(N_p-1):
            h_1 = np.vstack((copy.deepcopy(h_1), copy.deepcopy(action_max)))
        h_2 = copy.deepcopy(-action_min) 
        for i_7 in range(N_p-1):
            h_2 = np.vstack((copy.deepcopy(h_2), copy.deepcopy(-action_min)))
        
        # print(h_2)
            
        h_cvx = np.vstack((copy.deepcopy(h_1), copy.deepcopy(h_2)))
        P_cvx_ = matrix(copy.deepcopy(P_cvx))
        q_cvx_ = matrix(copy.deepcopy(q_cvx))
        G_cvx_ = matrix(copy.deepcopy(G_cvx))
        h_cvx_ = matrix(copy.deepcopy(h_cvx))
        

        
        result = 0
        result = copy.deepcopy(solvers.qp(copy.deepcopy(P_cvx_),copy.deepcopy(q_cvx_),copy.deepcopy(G_cvx_),copy.deepcopy(h_cvx_)))
        
        
        return result['x']
                    





def run(env, agent):
    Return = []
    E=[]
    target = 5
    step = 0
    

   
    x_a = []
    y_a = []
    x_b = []
    y_b = [] 
          
    ep=0
    action_mean = (env.action_space.high - env.action_space.low) / 2
    action_bias = (env.action_space.high + env.action_space.low) / 2
    
    state_mean = (env.observation_space.high - env.observation_space.low)/2
    state_bias = (env.observation_space.high + env.observation_space.low)/2

    
    for episode in range(config['max_episode']):
        ep+=1
     
        E.append(episode)
        
        
        # x_a = []
        # y_a = []
        # x_b = []
        # y_b = [] 
        

        
        
        score = 0
        state = env.reset(episode)


        
        
        #加载已经训练好的网络作为对方的策略
 
      
        # if episode > 50000:
        #     sac_b = SAC(env)
        #     sac_b.load_nets('./',39999)  
            
        for i in range(config['max_step']):
            
            # action_sac = agent.get_action(state) 
            
            # if episode < 2000 :




            #     # MPC控制——————————————————————————————————————————————————————————————
            #     x = state[0]
            #     y = state[1]
            #     pusin = state[2]
            #     v = state[3]
                 
            #     x_r = state[4] - 12.5*math.cos(state[6])
            #     y_r = state[5] - 12.5*math.sin(state[6])
    
            #     pusin_r = state[6]
            #     v_r = state[7]
                
                
            #     l_r = l_f = 2.5
            #     T = 0.1
            #     a_max = 10
            #     delta_max = np.pi/4
            #     a_min = -10
            #     delta_min = -np.pi/4 
            #     N_p=10
                
                
            #     action_MPC = 0
            #     s_mpc = MPC_QP_solver()
            #     action_MPC = s_mpc.sol(x,y,pusin,v,x_r,y_r,v_r,pusin_r,l_r,l_f,T,a_max,delta_max,a_min,delta_min,N_p)
                
            #     # print(action_MPC)
            #     action = [copy.deepcopy(action_MPC[0]),copy.deepcopy(action_MPC[1])]
            #     # print(action)
            #     action_mpc = (action - action_bias)/action_mean
                
            #     # if i>=300:
            #     #     action = [action_MPC[0],0]
            #     # ————————-—————————————————————————————————————————————————————————————— 
                
                
                
            #     action = action_mpc
            #     # print(action)
                





            # elif episode >= 2000:
            #     action = agent.get_action(state)

                
            # elif episode >=2000:   
            state_norm = (state - state_bias) / state_mean
            action = agent.get_action(state_norm)

            action_2 = [0,0]
            
            # if episode > 50000:
                
            #     # s_1 = state[4:8]
            #     # s_2 = state[0:4]
            #     # s_3 = state[8:10]       
            #     # a=np.hstack((s_1,s_2))
            #     # state_2=np.hstack((a,s_3))
            #     # action_2 = sac_b.get_action_test(state_2)

            #     #让敌方动作选择按照已经训练好的网络进行
            #     s_1 = state[4:8]
            #     s_2 = state[0:4]
            #     s_3 = state[8:9]
                
            #     a=np.hstack((s_1,s_2))            
            #     b=np.hstack((a,s_3))            
            #     s_4 = [env.get_min_m(state[4],state[5],state[6],state[7])]
            #     state_2=np.hstack((b,s_4))  
            #     action_2 = sac_b.get_action_test(state_2) 
                 
            
            next_state, reward, done, _ = env.step(action * action_mean + action_bias, action_2 * action_mean + action_bias,episode)
            done_mask = 0 if done else 1


            next_state_norm = (next_state - state_bias) / state_mean
            
            
            agent.push((state_norm, action, reward, next_state_norm, done_mask))
            state = next_state
            
             

            if episode+1==config['max_episode']:
                
                
                x_a.append(state[0])
                y_a.append(state[1])
                x_b.append(state[4])
                y_b.append(state[5]) 
                
  
            
            
            score += reward
            step +=1
            if done:
                break

            # if episode+1==config['max_episode']:
                
                
            #     x_a.append(state[0])
            #     y_a.append(state[1])
            #     x_b.append(state[4])
            #     y_b.append(state[5]) 
            
            
            if agent.buffer.buffer_len() > 2000:
                agent.update()
                
        # #画动态轨迹图    
        # if episode+1==config['max_episode']:
            
            
        #     plt.ion()  
        #     # plt.figure(1)
        #     #o_x, o_y = env.get_obstacle() 
        #     # plt.plot(o_x,o_y,'o')
            
            
        #     for j in range(0,len(o_x)):
                
        #         x_r, y_r = (o_x[j],o_y[j])
        #         # 半径
        #         r = 8
        #         #圆心坐标
        #         x2 = x_r
        #         y2 = y_r
        #         theta = np.arange(0, 2*np.pi, 0.01)
        #         x = x_r + r * np.cos(theta)
        #         y = y_r + r * np.sin(theta)
        #         fig = plt.figure(1)
        #         axes = fig.add_subplot(111) 
        #         axes.plot(x, y,c='g')
        #         plt.fill_between(x,y,y2,where=y>=y2,facecolor='green',interpolate=True)
        #         plt.fill_between(x,y,y2,where=y<=y2,facecolor='green',interpolate=True)
        #         # plt.fill_between(x,y,x2,where=x>=x2,facecolor='green',interpolate=True)
        #         # plt.fill_between(x,y,x2,where=x<=x2,facecolor='green',interpolate=True)
        #         axes.axis('equal')            
            
            
            
        #     for i in range(1, len(x_a)):
        #         ix_a = x_a[:i]
        #         iy_a = y_a[:i]
        #         ix_b = x_b[:i]
        #         iy_b = y_b[:i]
        #         plt.title("Trajectory")
        #         plt.plot(ix_a, iy_a, color='blue', label='car1')
        #         plt.plot(ix_b, iy_b, color='red', label='car2')
     
        #         plt.xlabel("x")
        #         plt.ylabel("y")
        #         ax=plt.gca()
        #         ax.xaxis.set_major_locator(MultipleLocator(5))
        #         ax.yaxis.set_major_locator(MultipleLocator(5))

        #         plt.xlim(round(min(min(x_a),min(x_b)))-1,math.ceil(max(max(x_a),max(x_b)))) #固定x轴
        #         plt.ylim(round(min(min(y_a),min(y_b)))-1,math.ceil(max(max(y_a),max(y_b)))) #固定y轴
        #         #plt.axis('equal')


        #         if i == 1:
        #             plt.pause(1)  # 启动时间
        #         plt.pause(0.5)

        #     plt.ioff()
        #     plt.show()                
           
        
       
        
        
        #   #画静态轨迹               
        # if episode+1==config['max_episode']:
            
        #     # plt.figure(2)
        #     # o_x, o_y = env.get_obstacle() 
        #     #plt.plot(o_x,o_y,'o')
        #     fig,ax=plt.subplots()            
        #     for j in range(0,len(o_x)):
                
        #         x_r, y_r = (o_x[j],o_y[j])
                                
        #         cir1 = Circle(xy = (x_r, y_r), radius=8, facecolor= 'black', alpha=0.5)
        #         ax.add_patch(cir1)
                                              
        
        #         plt.axis('scaled')
        #         ax.axis('equal')

            
        #     plt.title('Trajectory')
        #     plt.plot(x_a, y_a, color='blue', label='car1')
        #     plt.plot(x_b, y_b, color='red', label='car2')
        #     plt.legend()
        #     plt.xlabel('x')
        #     plt.ylabel('y')
        #     plt.show()

            
            
        # if episode%target==0:
        #     agent.save_nets('./',episode)
        
        
        
        
        if (episode+1)%5000==0:
            agent.save_nets('./',episode)        
                       
        if episode+1==config['max_episode']:
            agent.save_nets('./',episode)                      
   

        Return.append(score)
        print("episode:{}, Return:{}, Buffer_len:{}".format(episode, score, agent.buffer.buffer_len()))

        writer1.add_scalar('train_1/step_reward',score, step)
        writer2.add_scalar('train_2/episode_reward',score, ep)
    print("=======================end=======================")
    plt.figure(3)
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
env = CarCombatEnv()
sac = SAC(env)


run(env,sac)