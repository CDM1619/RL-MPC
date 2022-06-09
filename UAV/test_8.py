# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 18:59:07 2021

@author: 86153
"""
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import warnings
warnings.filterwarnings("ignore")
from matplotlib.patches import Circle
import torch
import os
import numpy as np
import random
import gym
import math
from matplotlib.pyplot import MultipleLocator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
# from AirCombat_test import AirCombatEnv
from AC_env_test import AirCombatEnv
from SAC_8 import config,SAC
import imageio,os
from matplotlib.animation import FuncAnimation
from math import sqrt
import time

i_1=0
j_1=0
def run(env, agent):
    E =[]
    R = []
    action_mean = (env.action_space.high - env.action_space.low) / 2
    action_bias = (env.action_space.high + env.action_space.low) / 2
    state_mean = (env.state_space.high - env.state_space.low) / 2
    state_bias = (env.state_space.high + env.state_space.low) / 2
    
    for episode in range(Episode):
        E.append(episode)
        x_a = []
        y_a = []
        z_a = []
        pusin_a = []
        gamma_a = []
        mju_a = []
        
        x_b = []
        y_b = []
        z_b = []
        pusin_b = []
        gamma_b = []
        mju_b = []
        
        score = 0
        state, state_2= env.reset()

        for i in range(step):
            state_norm = (state - state_bias)/state_mean
            action = agent.get_action_test(state_norm)
            next_state,next_state_2,reward, done, _, mju_b_ = env.step(action * action_mean + action_bias)
            # print(action)
            # next_state, reward, done, _ = env.step(action)

            state = next_state
            state_2 = next_state_2

            score += reward
            
            
            # if done:
            #     break
            # x_a.append(state[0])
            # y_a.append(state[1])
            # z_a.append(state[2])
            # x_b.append(state[6])
            # y_b.append(state[7])
            # z_b.append(state[8])   
            
            x_a.append(state_2[0])
            y_a.append(state_2[1])
            z_a.append(state_2[2])
            pusin_a.append(state_2[3])
            gamma_a.append(state_2[4])
            mju_a.append((action * action_mean + action_bias)[2])
            
            x_b.append(state_2[6])
            y_b.append(state_2[7])
            z_b.append(state_2[8])
            pusin_b.append(state_2[9])
            gamma_b.append(state_2[10])
            mju_b.append(mju_b_)



        if episode+1 == Episode: 
            global X_a,Y_a,Z_a,X_b,Y_b,Z_b
            X_a=x_a
            Y_a=y_a
            Z_a=z_a
            X_b=x_b
            Y_b=y_b
            Z_b=z_b

             
        
         
        #画静态轨迹
        if episode+1 == Episode:
            fig = plt.figure(2)
            ax1 = plt.axes(projection='3d')
            ax1.plot3D(x_a, y_a, z_a, 'blue')
            ax1.plot3D(x_b, y_b, z_b, 'red')      
            plt.savefig('./graph/paper/Trajectory.png',bbox_inches = 'tight',dpi=300)

            plt.show()







        #画静态轨迹------------------------------------------------------------------------------------------
        if episode+1 == Episode:

            #新建一个文件夹用来存放数据          
            path = "./graph/paper/Good Result/"
            # 定义文件夹名称
            name = "test_result_"         
            i_=0 
            
            while True:
                
                isExists = os.path.exists(path+name+str(i_))
                isExists_next = os.path.exists(path+name+str(i_+1))
                
                if not isExists:
                    os.makedirs(path+name+str(i_)) 
                    j_=i_                  
                    break
                elif isExists and not isExists_next:
                    os.makedirs(path+name+str(i_+1))
                    j_=i_+1
                    break
                elif isExists and isExists:
                    i_+=1
                    continue            


            fig = plt.figure(2)
            
            # ax = plt.axes(projection='3d')
            
   
            fig = plt.figure(figsize=(20,14),facecolor='white')  
            ax = fig.gca(fc='whitesmoke',projection='3d')
            
            ax.plot3D(x_a, y_a, z_a, 'blue')
            ax.plot3D(x_b, y_b, z_b, 'red')  

            for j in range(2):
                l1=40   #飞机前半身长度
                l2=30   #飞机后半身长度
                l3=30   #飞机宽度
                k=4     #绘制飞机的数量=k-1
                if j == 0:
                    color='blue'
                    
                for i in range(1,k):
                    x=x_a[len(x_a)//k*i]
                    y=y_a[len(x_a)//k*i]
                    z=z_a[len(x_a)//k*i]
                    pusin=pusin_a[len(x_a)//k*i]
                    gamma=gamma_a[len(x_a)//k*i]
                    mju=mju_a[len(x_a)//k*i]
    
    
                    
                    p1=[x+l1*math.cos(gamma)*math.sin(pusin),y+l1*math.cos(gamma)*math.cos(pusin),z+l1*math.sin(gamma)]
                    
                    p2=[x-l3/7*2*math.cos(pusin), y+l3/7*2*math.sin(pusin), z-l3/7*2*math.sin(mju)]
                    p23=[x+l3/7*2*math.cos(pusin), y-l3/7*2*math.sin(pusin), z+l3/7*2*math.sin(mju)]
                    
                    p3=[x-l2/5*1*math.cos(gamma)*math.sin(pusin)-l3/7*3*math.cos(pusin), y-l2/5*1*math.cos(gamma)*math.cos(pusin)+l3/7*3*math.sin(pusin), z-l2/5*1*math.sin(gamma)-l3/7*3*math.sin(mju)]
                    p22=[x-l2/5*1*math.cos(gamma)*math.sin(pusin)+l3/7*3*math.cos(pusin), y-l2/5*1*math.cos(gamma)*math.cos(pusin)-l3/7*3*math.sin(pusin), z-l2/5*1*math.sin(gamma)+l3/7*3*math.sin(mju)]
                    
                    p4=[x+l2/5*1*math.cos(gamma)*math.sin(pusin)-l3/2*1*math.cos(pusin), y+l2/5*1*math.cos(gamma)*math.cos(pusin)+l3/2*1*math.sin(pusin), z+l2/5*1*math.sin(gamma)-l3/2*1*math.sin(mju)]
                    p21=[x+l2/5*1*math.cos(gamma)*math.sin(pusin)+l3/2*1*math.cos(pusin), y+l2/5*1*math.cos(gamma)*math.cos(pusin)-l3/2*1*math.sin(pusin), z+l2/5*1*math.sin(gamma)+l3/2*1*math.sin(mju)]
                    
                    p5=[x-l2/5*2*math.cos(gamma)*math.sin(pusin)-l3/7*4*math.cos(pusin), y-l2/5*2*math.cos(gamma)*math.cos(pusin)+l3/7*4*math.sin(pusin), z-l2/5*2*math.sin(gamma)-l3/7*4*math.sin(mju)]
                    p20=[x-l2/5*2*math.cos(gamma)*math.sin(pusin)+l3/7*4*math.cos(pusin), y-l2/5*2*math.cos(gamma)*math.cos(pusin)-l3/7*4*math.sin(pusin), z-l2/5*2*math.sin(gamma)+l3/7*4*math.sin(mju)]
                    
                    p6=[x-l2/5*3*math.cos(gamma)*math.sin(pusin)-l3/7*5*math.cos(pusin), y-l2/5*3*math.cos(gamma)*math.cos(pusin)+l3/7*5*math.sin(pusin), z-l2/5*3*math.sin(gamma)-l3/7*5*math.sin(mju)]
                    p19=[x-l2/5*3*math.cos(gamma)*math.sin(pusin)+l3/7*5*math.cos(pusin), y-l2/5*3*math.cos(gamma)*math.cos(pusin)-l3/7*5*math.sin(pusin), z-l2/5*3*math.sin(gamma)+l3/7*5*math.sin(mju)]
                    
                    p7=[x-l3/14*11*math.cos(pusin), y+l3/14*11*math.sin(pusin), z-l3/14*11*math.sin(mju)]
                    p18=[x+l3/14*11*math.cos(pusin), y-l3/14*11*math.sin(pusin), z+l3/14*11*math.sin(mju)]
                    
                    p8=[x-l2/5*4*math.cos(gamma)*math.sin(pusin)-l3/7*6*math.cos(pusin), y-l2/5*4*math.cos(gamma)*math.cos(pusin)+l3/7*6*math.sin(pusin), z-l2/5*4*math.sin(gamma)-l3/7*6*math.sin(mju)]
                    p17=[x-l2/5*4*math.cos(gamma)*math.sin(pusin)+l3/7*6*math.cos(pusin), y-l2/5*4*math.cos(gamma)*math.cos(pusin)-l3/7*6*math.sin(pusin), z-l2/5*4*math.sin(gamma)+l3/7*6*math.sin(mju)]
                    
                    p9=[x-l2/5*5*math.cos(gamma)*math.sin(pusin)-l3/7*7*math.cos(pusin), y-l2/5*5*math.cos(gamma)*math.cos(pusin)+l3/7*7*math.sin(pusin), z-l2/5*5*math.sin(gamma)-l3/7*7*math.sin(mju)]
                    p16=[x-l2/5*5*math.cos(gamma)*math.sin(pusin)+l3/7*7*math.cos(pusin), y-l2/5*5*math.cos(gamma)*math.cos(pusin)-l3/7*7*math.sin(pusin), z-l2/5*5*math.sin(gamma)+l3/7*7*math.sin(mju)]
                    
                    p10=[x-l2/5*5*math.cos(gamma)*math.sin(pusin)-l3/7*1*math.cos(pusin), y-l2/5*5*math.cos(gamma)*math.cos(pusin)+l3/7*1*math.sin(pusin), z-l2/5*5*math.sin(gamma)-l3/7*1*math.sin(mju)]
                    p15=[x-l2/5*5*math.cos(gamma)*math.sin(pusin)+l3/7*1*math.cos(pusin), y-l2/5*5*math.cos(gamma)*math.cos(pusin)-l3/7*1*math.sin(pusin), z-l2/5*5*math.sin(gamma)+l3/7*1*math.sin(mju)]
                    
                    p11=[x-l2/5*6*math.cos(gamma)*math.sin(pusin)-l3/7*1*math.cos(pusin), y-l2/5*6*math.cos(gamma)*math.cos(pusin)+l3/7*1*math.sin(pusin), z-l2/5*6*math.sin(gamma)-l3/7*1*math.sin(mju)]
                    p14=[x-l2/5*6*math.cos(gamma)*math.sin(pusin)+l3/7*1*math.cos(pusin), y-l2/5*6*math.cos(gamma)*math.cos(pusin)-l3/7*1*math.sin(pusin), z-l2/5*6*math.sin(gamma)+l3/7*1*math.sin(mju)]
                    
                    p12=[x-l2/5*7*math.cos(gamma)*math.sin(pusin)-l3/7*2*math.cos(pusin), y-l2/5*7*math.cos(gamma)*math.cos(pusin)+l3/7*2*math.sin(pusin), z-l2/5*7*math.sin(gamma)-l3/7*2*math.sin(mju)]
                    p13=[x-l2/5*7*math.cos(gamma)*math.sin(pusin)+l3/7*2*math.cos(pusin), y-l2/5*7*math.cos(gamma)*math.cos(pusin)-l3/7*2*math.sin(pusin), z-l2/5*7*math.sin(gamma)+l3/7*2*math.sin(mju)]
                    
                
                    ax.plot3D(xs=[p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0], p9[0], p10[0], p11[0], p12[0], p13[0], p14[0], p15[0], p16[0], p17[0], p18[0], p19[0], p20[0], p21[0], p22[0], p23[0], p1[0],x],    # x 轴坐标
                              ys=[p1[1], p2[1], p3[1], p4[1], p5[1], p6[1], p7[1], p8[1], p9[1], p10[1], p11[1], p12[1], p13[1], p14[1], p15[1], p16[1], p17[1], p18[1], p19[1], p20[1], p21[1], p22[1], p23[1], p1[1],y],    # y 轴坐标
                              zs=[p1[2], p2[2], p3[2], p4[2], p5[2], p6[2], p7[2], p8[2], p9[2], p10[2], p11[2], p12[2], p13[2], p14[2], p15[2], p16[2], p17[2], p18[2], p19[2], p20[2], p21[2], p22[2], p23[2], p1[2],z],    # z 轴坐标
                              zdir='z',    # 
                              c=color,    # color
                              marker='o',    # 标记点符号
                              mfc=color,    # marker facecolor
                              mec=color,    # marker edgecolor
                              ms=1,    # size
                            )

                if j == 1:
                    color='red'

                for i in range(1,k):
                    x=x_b[len(x_b)//k*i]
                    y=y_b[len(x_b)//k*i]
                    z=z_b[len(x_b)//k*i]
                    pusin=pusin_b[len(x_b)//k*i]
                    gamma=gamma_b[len(x_b)//k*i]
                    mju=mju_b[len(x_b)//k*i]
    
    
                    
                    p1=[x+l1*math.cos(gamma)*math.sin(pusin),y+l1*math.cos(gamma)*math.cos(pusin),z+l1*math.sin(gamma)]
                    
                    p2=[x-l3/7*2*math.cos(pusin), y+l3/7*2*math.sin(pusin), z-l3/7*2*math.sin(mju)]
                    p23=[x+l3/7*2*math.cos(pusin), y-l3/7*2*math.sin(pusin), z+l3/7*2*math.sin(mju)]
                    
                    p3=[x-l2/5*1*math.cos(gamma)*math.sin(pusin)-l3/7*3*math.cos(pusin), y-l2/5*1*math.cos(gamma)*math.cos(pusin)+l3/7*3*math.sin(pusin), z-l2/5*1*math.sin(gamma)-l3/7*3*math.sin(mju)]
                    p22=[x-l2/5*1*math.cos(gamma)*math.sin(pusin)+l3/7*3*math.cos(pusin), y-l2/5*1*math.cos(gamma)*math.cos(pusin)-l3/7*3*math.sin(pusin), z-l2/5*1*math.sin(gamma)+l3/7*3*math.sin(mju)]
                    
                    p4=[x+l2/5*1*math.cos(gamma)*math.sin(pusin)-l3/2*1*math.cos(pusin), y+l2/5*1*math.cos(gamma)*math.cos(pusin)+l3/2*1*math.sin(pusin), z+l2/5*1*math.sin(gamma)-l3/2*1*math.sin(mju)]
                    p21=[x+l2/5*1*math.cos(gamma)*math.sin(pusin)+l3/2*1*math.cos(pusin), y+l2/5*1*math.cos(gamma)*math.cos(pusin)-l3/2*1*math.sin(pusin), z+l2/5*1*math.sin(gamma)+l3/2*1*math.sin(mju)]
                    
                    p5=[x-l2/5*2*math.cos(gamma)*math.sin(pusin)-l3/7*4*math.cos(pusin), y-l2/5*2*math.cos(gamma)*math.cos(pusin)+l3/7*4*math.sin(pusin), z-l2/5*2*math.sin(gamma)-l3/7*4*math.sin(mju)]
                    p20=[x-l2/5*2*math.cos(gamma)*math.sin(pusin)+l3/7*4*math.cos(pusin), y-l2/5*2*math.cos(gamma)*math.cos(pusin)-l3/7*4*math.sin(pusin), z-l2/5*2*math.sin(gamma)+l3/7*4*math.sin(mju)]
                    
                    p6=[x-l2/5*3*math.cos(gamma)*math.sin(pusin)-l3/7*5*math.cos(pusin), y-l2/5*3*math.cos(gamma)*math.cos(pusin)+l3/7*5*math.sin(pusin), z-l2/5*3*math.sin(gamma)-l3/7*5*math.sin(mju)]
                    p19=[x-l2/5*3*math.cos(gamma)*math.sin(pusin)+l3/7*5*math.cos(pusin), y-l2/5*3*math.cos(gamma)*math.cos(pusin)-l3/7*5*math.sin(pusin), z-l2/5*3*math.sin(gamma)+l3/7*5*math.sin(mju)]
                    
                    p7=[x-l3/14*11*math.cos(pusin), y+l3/14*11*math.sin(pusin), z-l3/14*11*math.sin(mju)]
                    p18=[x+l3/14*11*math.cos(pusin), y-l3/14*11*math.sin(pusin), z+l3/14*11*math.sin(mju)]
                    
                    p8=[x-l2/5*4*math.cos(gamma)*math.sin(pusin)-l3/7*6*math.cos(pusin), y-l2/5*4*math.cos(gamma)*math.cos(pusin)+l3/7*6*math.sin(pusin), z-l2/5*4*math.sin(gamma)-l3/7*6*math.sin(mju)]
                    p17=[x-l2/5*4*math.cos(gamma)*math.sin(pusin)+l3/7*6*math.cos(pusin), y-l2/5*4*math.cos(gamma)*math.cos(pusin)-l3/7*6*math.sin(pusin), z-l2/5*4*math.sin(gamma)+l3/7*6*math.sin(mju)]
                    
                    p9=[x-l2/5*5*math.cos(gamma)*math.sin(pusin)-l3/7*7*math.cos(pusin), y-l2/5*5*math.cos(gamma)*math.cos(pusin)+l3/7*7*math.sin(pusin), z-l2/5*5*math.sin(gamma)-l3/7*7*math.sin(mju)]
                    p16=[x-l2/5*5*math.cos(gamma)*math.sin(pusin)+l3/7*7*math.cos(pusin), y-l2/5*5*math.cos(gamma)*math.cos(pusin)-l3/7*7*math.sin(pusin), z-l2/5*5*math.sin(gamma)+l3/7*7*math.sin(mju)]
                    
                    p10=[x-l2/5*5*math.cos(gamma)*math.sin(pusin)-l3/7*1*math.cos(pusin), y-l2/5*5*math.cos(gamma)*math.cos(pusin)+l3/7*1*math.sin(pusin), z-l2/5*5*math.sin(gamma)-l3/7*1*math.sin(mju)]
                    p15=[x-l2/5*5*math.cos(gamma)*math.sin(pusin)+l3/7*1*math.cos(pusin), y-l2/5*5*math.cos(gamma)*math.cos(pusin)-l3/7*1*math.sin(pusin), z-l2/5*5*math.sin(gamma)+l3/7*1*math.sin(mju)]
                    
                    p11=[x-l2/5*6*math.cos(gamma)*math.sin(pusin)-l3/7*1*math.cos(pusin), y-l2/5*6*math.cos(gamma)*math.cos(pusin)+l3/7*1*math.sin(pusin), z-l2/5*6*math.sin(gamma)-l3/7*1*math.sin(mju)]
                    p14=[x-l2/5*6*math.cos(gamma)*math.sin(pusin)+l3/7*1*math.cos(pusin), y-l2/5*6*math.cos(gamma)*math.cos(pusin)-l3/7*1*math.sin(pusin), z-l2/5*6*math.sin(gamma)+l3/7*1*math.sin(mju)]
                    
                    p12=[x-l2/5*7*math.cos(gamma)*math.sin(pusin)-l3/7*2*math.cos(pusin), y-l2/5*7*math.cos(gamma)*math.cos(pusin)+l3/7*2*math.sin(pusin), z-l2/5*7*math.sin(gamma)-l3/7*2*math.sin(mju)]
                    p13=[x-l2/5*7*math.cos(gamma)*math.sin(pusin)+l3/7*2*math.cos(pusin), y-l2/5*7*math.cos(gamma)*math.cos(pusin)-l3/7*2*math.sin(pusin), z-l2/5*7*math.sin(gamma)+l3/7*2*math.sin(mju)]
                    
                
                    ax.plot3D(xs=[p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0], p9[0], p10[0], p11[0], p12[0], p13[0], p14[0], p15[0], p16[0], p17[0], p18[0], p19[0], p20[0], p21[0], p22[0], p23[0], p1[0],x],    # x 轴坐标
                              ys=[p1[1], p2[1], p3[1], p4[1], p5[1], p6[1], p7[1], p8[1], p9[1], p10[1], p11[1], p12[1], p13[1], p14[1], p15[1], p16[1], p17[1], p18[1], p19[1], p20[1], p21[1], p22[1], p23[1], p1[1],y],    # y 轴坐标
                              zs=[p1[2], p2[2], p3[2], p4[2], p5[2], p6[2], p7[2], p8[2], p9[2], p10[2], p11[2], p12[2], p13[2], p14[2], p15[2], p16[2], p17[2], p18[2], p19[2], p20[2], p21[2], p22[2], p23[2], p1[2],z],    # z 轴坐标
                              zdir='z',    # 
                              c=color,    # color
                              marker='o',    # 标记点符号
                              mfc=color,    # marker facecolor
                              mec=color,    # marker edgecolor
                              ms=1,    # size
                            )
                    


              # 设置坐标轴标题和刻度
            # left
            # ax.set(xlabel='X',
            #         ylabel='Y',
            #         zlabel='Z',
            #         xticks=np.arange(min(min(x_a),min(x_b)), max(max(x_a),max(x_b)), 1000),
            #         yticks=np.arange(min(min(y_a),min(y_b)), max(max(y_a),max(y_b)), 1000),
            #         zticks=np.arange(min(min(z_a),min(z_b)), max(max(z_a),max(z_b)), 1000)
            #       ) 
            ax.set(xlabel='X',
                    ylabel='Y',
                    zlabel='Z',
                  ) 
            # plt.axis('scaled')
            # ax.axis('equal') 
            # plt.gca().set_box_aspect((plt.xlim()[1]-plt.xlim()[0],plt.ylim()[1]-plt.ylim()[0],plt.zlim()[1]-plt.zlim()[0]))
       
            plt.legend(labels=['Trajectory of UCAV A','Trajectory of UCAV B'], loc='best')
            plt.savefig('./graph/paper/Good Result/test_result_'+str(j_)+'/Trajectory of UCAVs.png',bbox_inches = 'tight',dpi=300)
            plt.show()
        
            #----------------------------------------------------------------------------------------------






        # # 画动态态轨迹-----------------------------------------------------------------
        # if episode+1 == Episode:
            
            
        #     #新建一个文件夹用来存放数据          
        #     path = "./graph/paper/Good Result/"
        #     # 定义文件夹名称
        #     name = "test_result_" 
            
        #     i_=0 
          
        #     while True:
                
        #         isExists = os.path.exists(path+name+str(i_))
        #         isExists_next = os.path.exists(path+name+str(i_+1))
                
        #         if not isExists:
        #             os.makedirs(path+name+str(i_)) 
        #             # 创建一个新文件夹用来存放动图
        #             os.makedirs("./graph/paper/Good Result/test_result_"+str(i_)+"/Gif")   
        #             j_=i_                  
        #             break
        #         elif isExists and not isExists_next:
        #             os.makedirs(path+name+str(i_+1))
        #             # 创建一个新文件夹用来存放动图
        #             os.makedirs("./graph/paper/Good Result/test_result_"+str(i_+1)+"/Gif") 
        #             j_=i_+1
        #             break
        #         elif isExists and isExists:
        #             i_+=1
        #             continue

            
        #     # fig = plt.figure(2)
        #     # ax = plt.axes(projection='3d')
            
        #     fig = plt.figure(figsize=(20,14),facecolor='lightyellow')           
        #     ax = fig.gca(fc='whitesmoke',projection='3d')
            

                    
        #     for i in range(0,len(x_a)):
        #         plt.cla() 
        #         l1=40
        #         l2=30
        #         l3=30
        #         color='blue'
        #         x=x_a[i]
        #         y=y_a[i]
        #         z=z_a[i]
        #         pusin=pusin_a[i]
        #         gamma=gamma_a[i]
        #         mju=mju_a[i]


                
        #         p1=[x+l1*math.cos(gamma)*math.sin(pusin),y+l1*math.cos(gamma)*math.cos(pusin),z+l1*math.sin(gamma)]
                
        #         p2=[x-l3/7*2*math.cos(pusin), y+l3/7*2*math.sin(pusin), z-l3/7*2*math.sin(mju)]
        #         p23=[x+l3/7*2*math.cos(pusin), y-l3/7*2*math.sin(pusin), z+l3/7*2*math.sin(mju)]
                
        #         p3=[x-l2/5*1*math.cos(gamma)*math.sin(pusin)-l3/7*3*math.cos(pusin), y-l2/5*1*math.cos(gamma)*math.cos(pusin)+l3/7*3*math.sin(pusin), z-l2/5*1*math.sin(gamma)-l3/7*3*math.sin(mju)]
        #         p22=[x-l2/5*1*math.cos(gamma)*math.sin(pusin)+l3/7*3*math.cos(pusin), y-l2/5*1*math.cos(gamma)*math.cos(pusin)-l3/7*3*math.sin(pusin), z-l2/5*1*math.sin(gamma)+l3/7*3*math.sin(mju)]
                
        #         p4=[x+l2/5*1*math.cos(gamma)*math.sin(pusin)-l3/2*1*math.cos(pusin), y+l2/5*1*math.cos(gamma)*math.cos(pusin)+l3/2*1*math.sin(pusin), z+l2/5*1*math.sin(gamma)-l3/2*1*math.sin(mju)]
        #         p21=[x+l2/5*1*math.cos(gamma)*math.sin(pusin)+l3/2*1*math.cos(pusin), y+l2/5*1*math.cos(gamma)*math.cos(pusin)-l3/2*1*math.sin(pusin), z+l2/5*1*math.sin(gamma)+l3/2*1*math.sin(mju)]
                
        #         p5=[x-l2/5*2*math.cos(gamma)*math.sin(pusin)-l3/7*4*math.cos(pusin), y-l2/5*2*math.cos(gamma)*math.cos(pusin)+l3/7*4*math.sin(pusin), z-l2/5*2*math.sin(gamma)-l3/7*4*math.sin(mju)]
        #         p20=[x-l2/5*2*math.cos(gamma)*math.sin(pusin)+l3/7*4*math.cos(pusin), y-l2/5*2*math.cos(gamma)*math.cos(pusin)-l3/7*4*math.sin(pusin), z-l2/5*2*math.sin(gamma)+l3/7*4*math.sin(mju)]
                
        #         p6=[x-l2/5*3*math.cos(gamma)*math.sin(pusin)-l3/7*5*math.cos(pusin), y-l2/5*3*math.cos(gamma)*math.cos(pusin)+l3/7*5*math.sin(pusin), z-l2/5*3*math.sin(gamma)-l3/7*5*math.sin(mju)]
        #         p19=[x-l2/5*3*math.cos(gamma)*math.sin(pusin)+l3/7*5*math.cos(pusin), y-l2/5*3*math.cos(gamma)*math.cos(pusin)-l3/7*5*math.sin(pusin), z-l2/5*3*math.sin(gamma)+l3/7*5*math.sin(mju)]
                
        #         p7=[x-l3/14*11*math.cos(pusin), y+l3/14*11*math.sin(pusin), z-l3/14*11*math.sin(mju)]
        #         p18=[x+l3/14*11*math.cos(pusin), y-l3/14*11*math.sin(pusin), z+l3/14*11*math.sin(mju)]
                
        #         p8=[x-l2/5*4*math.cos(gamma)*math.sin(pusin)-l3/7*6*math.cos(pusin), y-l2/5*4*math.cos(gamma)*math.cos(pusin)+l3/7*6*math.sin(pusin), z-l2/5*4*math.sin(gamma)-l3/7*6*math.sin(mju)]
        #         p17=[x-l2/5*4*math.cos(gamma)*math.sin(pusin)+l3/7*6*math.cos(pusin), y-l2/5*4*math.cos(gamma)*math.cos(pusin)-l3/7*6*math.sin(pusin), z-l2/5*4*math.sin(gamma)+l3/7*6*math.sin(mju)]
                
        #         p9=[x-l2/5*5*math.cos(gamma)*math.sin(pusin)-l3/7*7*math.cos(pusin), y-l2/5*5*math.cos(gamma)*math.cos(pusin)+l3/7*7*math.sin(pusin), z-l2/5*5*math.sin(gamma)-l3/7*7*math.sin(mju)]
        #         p16=[x-l2/5*5*math.cos(gamma)*math.sin(pusin)+l3/7*7*math.cos(pusin), y-l2/5*5*math.cos(gamma)*math.cos(pusin)-l3/7*7*math.sin(pusin), z-l2/5*5*math.sin(gamma)+l3/7*7*math.sin(mju)]
                
        #         p10=[x-l2/5*5*math.cos(gamma)*math.sin(pusin)-l3/7*1*math.cos(pusin), y-l2/5*5*math.cos(gamma)*math.cos(pusin)+l3/7*1*math.sin(pusin), z-l2/5*5*math.sin(gamma)-l3/7*1*math.sin(mju)]
        #         p15=[x-l2/5*5*math.cos(gamma)*math.sin(pusin)+l3/7*1*math.cos(pusin), y-l2/5*5*math.cos(gamma)*math.cos(pusin)-l3/7*1*math.sin(pusin), z-l2/5*5*math.sin(gamma)+l3/7*1*math.sin(mju)]
                
        #         p11=[x-l2/5*6*math.cos(gamma)*math.sin(pusin)-l3/7*1*math.cos(pusin), y-l2/5*6*math.cos(gamma)*math.cos(pusin)+l3/7*1*math.sin(pusin), z-l2/5*6*math.sin(gamma)-l3/7*1*math.sin(mju)]
        #         p14=[x-l2/5*6*math.cos(gamma)*math.sin(pusin)+l3/7*1*math.cos(pusin), y-l2/5*6*math.cos(gamma)*math.cos(pusin)-l3/7*1*math.sin(pusin), z-l2/5*6*math.sin(gamma)+l3/7*1*math.sin(mju)]
                
        #         p12=[x-l2/5*7*math.cos(gamma)*math.sin(pusin)-l3/7*2*math.cos(pusin), y-l2/5*7*math.cos(gamma)*math.cos(pusin)+l3/7*2*math.sin(pusin), z-l2/5*7*math.sin(gamma)-l3/7*2*math.sin(mju)]
        #         p13=[x-l2/5*7*math.cos(gamma)*math.sin(pusin)+l3/7*2*math.cos(pusin), y-l2/5*7*math.cos(gamma)*math.cos(pusin)-l3/7*2*math.sin(pusin), z-l2/5*7*math.sin(gamma)+l3/7*2*math.sin(mju)]
                
            
        #         ax.plot3D(xs=[p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0], p9[0], p10[0], p11[0], p12[0], p13[0], p14[0], p15[0], p16[0], p17[0], p18[0], p19[0], p20[0], p21[0], p22[0], p23[0], p1[0],x],    # x 轴坐标
        #                   ys=[p1[1], p2[1], p3[1], p4[1], p5[1], p6[1], p7[1], p8[1], p9[1], p10[1], p11[1], p12[1], p13[1], p14[1], p15[1], p16[1], p17[1], p18[1], p19[1], p20[1], p21[1], p22[1], p23[1], p1[1],y],    # y 轴坐标
        #                   zs=[p1[2], p2[2], p3[2], p4[2], p5[2], p6[2], p7[2], p8[2], p9[2], p10[2], p11[2], p12[2], p13[2], p14[2], p15[2], p16[2], p17[2], p18[2], p19[2], p20[2], p21[2], p22[2], p23[2], p1[2],z],    # z 轴坐标
        #                   zdir='z',    # 
        #                   c=color,    # color
        #                   marker='o',    # 标记点符号
        #                   mfc=color,    # marker facecolor
        #                   mec=color,    # marker edgecolor
        #                   ms=1,    # size
        #                 )
                
                
                

        #         color='red'
        #         x=x_b[i]
        #         y=y_b[i]
        #         z=z_b[i]
        #         pusin=pusin_b[i]
        #         gamma=gamma_b[i]
        #         mju=mju_b[i]


        #         p1=[x+l1*math.cos(gamma)*math.sin(pusin),y+l1*math.cos(gamma)*math.cos(pusin),z+l1*math.sin(gamma)]
                
        #         p2=[x-l3/7*2*math.cos(pusin), y+l3/7*2*math.sin(pusin), z-l3/7*2*math.sin(mju)]
        #         p23=[x+l3/7*2*math.cos(pusin), y-l3/7*2*math.sin(pusin), z+l3/7*2*math.sin(mju)]
                
        #         p3=[x-l2/5*1*math.cos(gamma)*math.sin(pusin)-l3/7*3*math.cos(pusin), y-l2/5*1*math.cos(gamma)*math.cos(pusin)+l3/7*3*math.sin(pusin), z-l2/5*1*math.sin(gamma)-l3/7*3*math.sin(mju)]
        #         p22=[x-l2/5*1*math.cos(gamma)*math.sin(pusin)+l3/7*3*math.cos(pusin), y-l2/5*1*math.cos(gamma)*math.cos(pusin)-l3/7*3*math.sin(pusin), z-l2/5*1*math.sin(gamma)+l3/7*3*math.sin(mju)]
                
        #         p4=[x+l2/5*1*math.cos(gamma)*math.sin(pusin)-l3/2*1*math.cos(pusin), y+l2/5*1*math.cos(gamma)*math.cos(pusin)+l3/2*1*math.sin(pusin), z+l2/5*1*math.sin(gamma)-l3/2*1*math.sin(mju)]
        #         p21=[x+l2/5*1*math.cos(gamma)*math.sin(pusin)+l3/2*1*math.cos(pusin), y+l2/5*1*math.cos(gamma)*math.cos(pusin)-l3/2*1*math.sin(pusin), z+l2/5*1*math.sin(gamma)+l3/2*1*math.sin(mju)]
                
        #         p5=[x-l2/5*2*math.cos(gamma)*math.sin(pusin)-l3/7*4*math.cos(pusin), y-l2/5*2*math.cos(gamma)*math.cos(pusin)+l3/7*4*math.sin(pusin), z-l2/5*2*math.sin(gamma)-l3/7*4*math.sin(mju)]
        #         p20=[x-l2/5*2*math.cos(gamma)*math.sin(pusin)+l3/7*4*math.cos(pusin), y-l2/5*2*math.cos(gamma)*math.cos(pusin)-l3/7*4*math.sin(pusin), z-l2/5*2*math.sin(gamma)+l3/7*4*math.sin(mju)]
                
        #         p6=[x-l2/5*3*math.cos(gamma)*math.sin(pusin)-l3/7*5*math.cos(pusin), y-l2/5*3*math.cos(gamma)*math.cos(pusin)+l3/7*5*math.sin(pusin), z-l2/5*3*math.sin(gamma)-l3/7*5*math.sin(mju)]
        #         p19=[x-l2/5*3*math.cos(gamma)*math.sin(pusin)+l3/7*5*math.cos(pusin), y-l2/5*3*math.cos(gamma)*math.cos(pusin)-l3/7*5*math.sin(pusin), z-l2/5*3*math.sin(gamma)+l3/7*5*math.sin(mju)]
                
        #         p7=[x-l3/14*11*math.cos(pusin), y+l3/14*11*math.sin(pusin), z-l3/14*11*math.sin(mju)]
        #         p18=[x+l3/14*11*math.cos(pusin), y-l3/14*11*math.sin(pusin), z+l3/14*11*math.sin(mju)]
                
        #         p8=[x-l2/5*4*math.cos(gamma)*math.sin(pusin)-l3/7*6*math.cos(pusin), y-l2/5*4*math.cos(gamma)*math.cos(pusin)+l3/7*6*math.sin(pusin), z-l2/5*4*math.sin(gamma)-l3/7*6*math.sin(mju)]
        #         p17=[x-l2/5*4*math.cos(gamma)*math.sin(pusin)+l3/7*6*math.cos(pusin), y-l2/5*4*math.cos(gamma)*math.cos(pusin)-l3/7*6*math.sin(pusin), z-l2/5*4*math.sin(gamma)+l3/7*6*math.sin(mju)]
                
        #         p9=[x-l2/5*5*math.cos(gamma)*math.sin(pusin)-l3/7*7*math.cos(pusin), y-l2/5*5*math.cos(gamma)*math.cos(pusin)+l3/7*7*math.sin(pusin), z-l2/5*5*math.sin(gamma)-l3/7*7*math.sin(mju)]
        #         p16=[x-l2/5*5*math.cos(gamma)*math.sin(pusin)+l3/7*7*math.cos(pusin), y-l2/5*5*math.cos(gamma)*math.cos(pusin)-l3/7*7*math.sin(pusin), z-l2/5*5*math.sin(gamma)+l3/7*7*math.sin(mju)]
                
        #         p10=[x-l2/5*5*math.cos(gamma)*math.sin(pusin)-l3/7*1*math.cos(pusin), y-l2/5*5*math.cos(gamma)*math.cos(pusin)+l3/7*1*math.sin(pusin), z-l2/5*5*math.sin(gamma)-l3/7*1*math.sin(mju)]
        #         p15=[x-l2/5*5*math.cos(gamma)*math.sin(pusin)+l3/7*1*math.cos(pusin), y-l2/5*5*math.cos(gamma)*math.cos(pusin)-l3/7*1*math.sin(pusin), z-l2/5*5*math.sin(gamma)+l3/7*1*math.sin(mju)]
                
        #         p11=[x-l2/5*6*math.cos(gamma)*math.sin(pusin)-l3/7*1*math.cos(pusin), y-l2/5*6*math.cos(gamma)*math.cos(pusin)+l3/7*1*math.sin(pusin), z-l2/5*6*math.sin(gamma)-l3/7*1*math.sin(mju)]
        #         p14=[x-l2/5*6*math.cos(gamma)*math.sin(pusin)+l3/7*1*math.cos(pusin), y-l2/5*6*math.cos(gamma)*math.cos(pusin)-l3/7*1*math.sin(pusin), z-l2/5*6*math.sin(gamma)+l3/7*1*math.sin(mju)]
                
        #         p12=[x-l2/5*7*math.cos(gamma)*math.sin(pusin)-l3/7*2*math.cos(pusin), y-l2/5*7*math.cos(gamma)*math.cos(pusin)+l3/7*2*math.sin(pusin), z-l2/5*7*math.sin(gamma)-l3/7*2*math.sin(mju)]
        #         p13=[x-l2/5*7*math.cos(gamma)*math.sin(pusin)+l3/7*2*math.cos(pusin), y-l2/5*7*math.cos(gamma)*math.cos(pusin)-l3/7*2*math.sin(pusin), z-l2/5*7*math.sin(gamma)+l3/7*2*math.sin(mju)]
                
            
        #         ax.plot3D(xs=[p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0], p9[0], p10[0], p11[0], p12[0], p13[0], p14[0], p15[0], p16[0], p17[0], p18[0], p19[0], p20[0], p21[0], p22[0], p23[0], p1[0],x],    # x 轴坐标
        #                   ys=[p1[1], p2[1], p3[1], p4[1], p5[1], p6[1], p7[1], p8[1], p9[1], p10[1], p11[1], p12[1], p13[1], p14[1], p15[1], p16[1], p17[1], p18[1], p19[1], p20[1], p21[1], p22[1], p23[1], p1[1],y],    # y 轴坐标
        #                   zs=[p1[2], p2[2], p3[2], p4[2], p5[2], p6[2], p7[2], p8[2], p9[2], p10[2], p11[2], p12[2], p13[2], p14[2], p15[2], p16[2], p17[2], p18[2], p19[2], p20[2], p21[2], p22[2], p23[2], p1[2],z],    # z 轴坐标
        #                   zdir='z',    # 
        #                   c=color,    # color
        #                   marker='o',    # 标记点符号
        #                   mfc=color,    # marker facecolor
        #                   mec=color,    # marker edgecolor
        #                   ms=1,    # size
        #                 )
        #         ax.plot3D(x_a[:i+1], y_a[:i+1], z_a[:i+1], 'blue')
        #         ax.plot3D(x_b[:i+1], y_b[:i+1], z_b[:i+1], 'red')  
        #         ax.set(xlabel='X',
        #             ylabel='Y',
        #             zlabel='Z',
        #           ) 
        #         plt.savefig('./graph/paper/Good Result/test_result_'+str(j_)+'/Gif/Trajectory%d.png'%i,bbox_inches = 'tight',dpi=300)

        #     #dpi合成gif
            
        #     fig_ = []
        #     for i in range(0, len(x_a)-1):
        #         # fig.append(imageio.imread("Trajectory" + str(i) + ".png"))
        #         fig_.append(imageio.imread("./graph/paper/Good Result/test_result_"+str(j_)+"/Gif/Trajectory" + str(i) + ".png"))    
        #         # fig.append(imageio.imread("./graph/paper/Good Result/test_result_21/Gif/Trajectory" + str(i) + ".png"))    
              
        #     gif_1 = "./graph/paper/Good Result/test_result_"+str(j_)+"/Gif/Trajectory.gif"
        #     imageio.mimsave(gif_1, fig_, duration = 0.001)  
            
        #     #-----------------------------------------------------------------------------------------------          
            






        R.append(score)
        print("episode:{}, Return:{}, Buffer_len:{}".format(episode, score, agent.buffer.buffer_len()))
        env.close()
    print("=======================end=======================")
    # plt.figure(3)
    # plt.plot(E,R)
    # plt.legend()
    # plt.show()

Episode =1
step = 1000
    # env and RL param
torch.set_num_threads(config['num_threads'])
os.environ['MKL_NUM_THREADS'] = str(config['num_threads'])
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
if not config['random_seed']:
    torch.cuda.manual_seed_all(config['seed'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
# method

start = time.time()
env = AirCombatEnv()
sac = SAC(env)
sac.load_nets('./',23999)

run(env,sac)
end = time.time()
print(end-start)

make_gif=0

if make_gif:
        

    #绘制动态轨迹图
    
    fig=plt.figure(figsize=(10,10))
    ax1 = Axes3D(fig)
    ax1.plot3D([0,1],[0,1],[0,1], 'red')
    
    j=0
    i=0
    
    a_track = np.array([[X_a[0],Y_a[0],Z_a[0]]])
    a_track_s = np.array([.0,.0,.0])
    
    b_track = np.array([[X_b[0],Y_b[0],Z_b[0]]])
    b_track_s = np.array([.0,.0,.0])
    
    
    def gen_path():
        global a_track_s,a_track,j,b_track_s,b_track
        j+=1
        
     
        x_1 = 0
        y_1 = 0
        z_1 = 0
        x_2 = 0
        y_2 = 0
        z_2 = 0
    
            
        
        x_1 = X_a[j]
        y_1 = Y_a[j]
        z_1 = Z_a[j]
        
        x_2 = X_b[j]
        y_2 = Y_b[j]
        z_2 = Z_b[j] 
    
            
    
    
        
        a_track_s =[x_1,y_1,z_1]
    
        a_track = np.append(a_track, [a_track_s],axis=0)
        
    
            
        b_track_s =[x_2,y_2,z_2]
    
        b_track = np.append(b_track, [b_track_s],axis=0)    
        
        return a_track, b_track
    
    def update(i):
        
        #label = 'timestep {0}'.format(i)
        #print("step:",i)
            
        i+=1
        # 更新直线和x轴（用一个新的x轴的标签）。
        # 用元组（Tuple）的形式返回在这一帧要被重新绘图的物体
        a_track,b_track = gen_path()
        # ax1.set_xlabel(label)
    
        ax1.plot3D(a_track[:, 0], a_track[:, 1], a_track[:, 2], 'blue')
        ax1.plot3D(b_track[:, 0], b_track[:, 1], b_track[:, 2], 'red')
        
        
        ax1.set_xlabel("X Axis")
        ax1.set_ylabel("Y Axis")
        ax1.set_zlabel("Z Axis")
        ax1.set_title("Trajectory")
        ax1.set(xlim=[round(min(min(X_a),min(X_b)))-1,math.ceil(max(max(X_a),max(X_b)))], 
                ylim=[round(min(min(Y_a),min(Y_b)))-1,math.ceil(max(max(Y_a),max(Y_b)))], 
                zlim=[round(min(min(Z_a),min(Z_b)))-1,math.ceil(max(max(Z_a),max(Z_b)))])      
        
        
        return ax1
    
            
    anim = FuncAnimation(fig, update, frames=np.arange(0,len(X_a)), interval=0.01, repeat = False)
    # anim = FuncAnimation(fig, update, frames=len(X_a), interval=0.1, repeat=False)
        
    anim.save('./graph/paper/b.gif')  
    # plt.show()
    

