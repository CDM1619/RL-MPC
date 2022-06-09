# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 18:24:08 2021

@author: 86153
"""

#动态避障（障碍物位置会变化）

from PIL import Image
import sys
import imageio,os
import copy
from openpyxl import Workbook
from math import sqrt, pow
import torch
import os
import numpy as np
import random
import gym
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import warnings
warnings.filterwarnings("ignore")
from matplotlib.patches import Circle
import copy
from matplotlib.animation import FuncAnimation

from SAC import config,SAC
from CarCombat_dyna_avoid_2_test import CarCombatEnv

def run(env, agent):
    E =[]
    R = []
    R_b = []
    
    R__ = []
    R_b__ = []  
    R_error=[]
    
    
    #用来计算N回合中平均胜率
    Win_Total_A=[]    
    Win_Total_B=[]   
    Episode_Adv_A=[]
    Episode_Adv_B=[]    
    for episode in range(Episode):
        E.append(episode)
        x_a = []
        y_a = []
        x_b = []
        y_b = []
        radi=[]
        
        O_x=[]
        O_y=[]
        
        p1_a=[]
        p1_b=[]

        

        score = 0


        
        state,o_x,o_y = env.reset()

        action_mean = (env.action_space.high - env.action_space.low) / 2
        action_bias = (env.action_space.high + env.action_space.low) / 2
        
        # B车采取的策略：用A车之前训练好的策略网络      
        sac_b = SAC(env)
        sac_b.load_nets('./',89999)
        
        for i in range(step):
            
           # #追踪和避障用两个网络  
            
           #  min_m = env.get_min_m(state[0],state[1])
           #  if min_m <= env.get_r()+10:
           #      agent = SAC(env)
           #      agent.load_nets('./',89999)
           #      action = agent.get_action_test(state)
           #  else:
           #      agent = SAC(env)
           #      agent.load_nets('./',19999)
           #      action = agent.get_action_test(state)
                
            
            action = agent.get_action_test(state)
            
               
            #让敌方动作选择按照已经训练好的网络进行
            s_1 = state[4:8]
            s_2 = state[0:4]
            s_3 = state[8:9]
            
            a=np.hstack((s_1,s_2))            
            b=np.hstack((a,s_3))            
            s_4 = [env.get_min_m(state[4],state[5],state[6],state[7])]
            state_2=np.hstack((b,s_4))  
            action_2 = sac_b.get_action_test(state_2)
            

            # #动态避障-----------------------------------------
            # global o_x
            # global o_y
            
            # o_x = []
            # o_y = []
            
            # global r
                
            # random.seed(i)
            # r=20
            # for k in range(0,30):
            #     o_x.append(random.uniform(-300,300))
            #     o_y.append(random.uniform(-300,300))   
            # next_state, reward, done, _, o_x, o_y, reward__ = env.step(action * action_mean + action_bias, action_2 * action_mean + action_bias,episode,i,o_x,o_y,r)
           
            # #--------------------------------------------
            
            
            
                 
            next_state, reward, done, _, next_o_x, next_o_y, reward__ = env.step(action * action_mean + action_bias, action_2 * action_mean + action_bias,episode,i,)
            
            reward_b__ = env.get_b_step_reward(state[0], state[1], state[2], state[3], state[4], state[5], state[6], state[7], o_x, o_y)             
            # radi.append(env.get_r()) 

                      
            
            
            # #让敌方动作选择按照随机给出
            # next_state, reward, done, _, o_x, o_y = env.step(action * action_mean + action_bias, 0,episode)
                  
            if done:
                break            
            d = env.get_distance()
            angle_1, angle_2 = env.get_angle()
            v_a = env.get_v_a()
            v_b = env.get_v_b()
            # o_x, o_y = env.get_obstacle() 
            



            

            state = next_state
            o_x = next_o_x
            o_y = next_o_y
            # print(o_y)
            o_x_=copy.deepcopy(o_x)    #注意这里要用深拷贝，不然列表会跟着变化
            o_y_=copy.deepcopy(o_y) 
            
            

            
            
            score = score + reward
            
            # 综合优势
            if d<=20 and angle_1<=50:
                p1_a.append(1)
                p1_b.append(-1)
                
            elif d<=20 and angle_2<=50:
                p1_a.append(-1)
                p1_b.append(1)
                
            else:
                p1_a.append(0)
                p1_b.append(0)
      
            
            x_a.append(state[0])
            y_a.append(state[1])
            x_b.append(state[4])
            y_b.append(state[5])
            
            O_x.append(o_x_)
            O_y.append(o_y_)

            # print(O_y)   
               
          #绘制动态轨迹  (有障碍物)   

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
                    # 创建一个新文件夹用来存放动图
                    os.makedirs("./graph/paper/Good Result/test_result_"+str(i_)+"/Gif")   
                    j_=i_                  
                    break
                elif isExists and not isExists_next:
                    os.makedirs(path+name+str(i_+1))
                    # 创建一个新文件夹用来存放动图
                    os.makedirs("./graph/paper/Good Result/test_result_"+str(i_+1)+"/Gif") 
                    j_=i_+1
                    break
                elif isExists and isExists:
                    i_+=1
                    continue
                                             
            
            fig,ax=plt.subplots()  
            
        
            
            
           #画矩形用来画小车           
            def angle_of_vector(v1, v2):
                pi = np.pi
                vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
                length_prod = sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * sqrt(pow(v2[0], 2) + pow(v2[1], 2))
                cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
                return (math.acos(cos) / pi) * 180            


            k=0
            

            # A车和B车的初始血量
            blood = 100.0
            w_a = blood
            w_b = blood
            # 总血量
            w = blood
            
            # 方案2：车身长度固定  
            for i in range(0,len(x_a)-1): 
                plt.cla()                    
                x1_a = x_a[i]
                y1_a = y_a[i] 
                
                x2_a = x_a[i+1]
                y2_a = y_a[i+1]
                           
                
                ang_a = angle_of_vector([x2_a-x1_a,y2_a-y1_a],[1,0])
    
                
                if x2_a-x1_a<0 and y2_a-y1_a<=0:
                    x1_a = x2_a - 27* math.cos(ang_a/180*np.pi)
                    y1_a = y2_a + 27* math.sin(ang_a/180*np.pi)
                                        
                    
                    
                    ang_aa = -ang_a
                    ang_a = 180-ang_a 


                    x2_aa = x1_a + 40* math.cos(ang_a/180*np.pi)
                    y2_aa = y1_a - 40* math.sin(ang_a/180*np.pi)
                    ax.arrow(x1_a,y1_a,-x2_aa+x1_a,y2_aa-y1_a,length_includes_head = True,head_width = 13,head_length = 13,fc='royalblue',ec='black',zorder=11)
 

                    
                elif x2_a-x1_a<=0 and y2_a-y1_a>0:
                    x1_a = x2_a - 27* math.cos(ang_a/180*np.pi)
                    y1_a = y2_a - 27* math.sin(ang_a/180*np.pi)                 
                    
                    
                    
                    ang_aa = ang_a
                    ang_a = ang_a-180 
                    
            
                    x2_aa = x1_a + 40* math.cos(ang_a/180*np.pi)
                    y2_aa = y1_a + 40* math.sin(ang_a/180*np.pi)          
                    ax.arrow(x1_a,y1_a,-x2_aa+x1_a,-y2_aa+y1_a,length_includes_head = True,head_width = 13,head_length = 13,fc='royalblue',ec='black',zorder=11)
                   
    
    
                    
                elif x2_a-x1_a>0 and y2_a-y1_a>=0:
                    x1_a = x2_a - 27* math.cos(ang_a/180*np.pi)
                    y1_a = y2_a - 27* math.sin(ang_a/180*np.pi)
                    
                    
                    ang_aa = ang_a
                    ang_a = ang_a-180 


                    x2_aa = x1_a + 40* math.cos(ang_a/180*np.pi)
                    y2_aa = y1_a + 40* math.sin(ang_a/180*np.pi)          
                    ax.arrow(x1_a,y1_a,-x2_aa+x1_a,-y2_aa+y1_a,length_includes_head = True,head_width = 13,head_length = 13,fc='royalblue',ec='black',zorder=11)
                        
    

    
                    
                elif x2_a-x1_a>=0 and y2_a-y1_a<0:
                    x1_a = x2_a - 27* math.cos(ang_a/180*np.pi)
                    y1_a = y2_a + 27* math.sin(ang_a/180*np.pi)                   
                    
                    
                    ang_aa = -ang_a
                    ang_a = 180-ang_a    


                    x2_aa = x1_a + 40* math.cos(ang_a/180*np.pi)
                    y2_aa = y1_a - 40* math.sin(ang_a/180*np.pi)          
                    # ax.arrow(x1_a,y1_a,-x2_aa+x1_a,y2_aa-y1_a,length_includes_head = True,head_width = 13,head_length = 13,fc='darkorange',ec='black',zorder=9)
                    ax.arrow(x1_a,y1_a,-x2_aa+x1_a,y2_aa-y1_a,length_includes_head = True,head_width = 13,head_length = 13,fc='royalblue',ec='black',zorder=11)  
                    
                       
                                            
                
                #车身 （长度固定）
                square1 = plt.Rectangle(xy=(x2_a, y2_a), width = 27, height = 7, angle=ang_a, color='royalblue',ec='black',zorder=12)                  
                              
                #车身
                square2 = plt.Rectangle(xy=(x2_a, y2_a), width = 27, height = -7, angle=ang_a, color='royalblue',ec='black',zorder=12)                  
    
                #前车轮                          
                square3 = plt.Rectangle(xy=(x2_a, y2_a), width = 5, height = 10, angle=ang_a, color='black',ec='black',zorder=11)                  
               
                #前车轮
                square4 = plt.Rectangle(xy=(x2_a, y2_a), width = 5, height = -10, angle=ang_a, color='black',ec='black',zorder=11)                  
     
                #后车轮                          
                square5 = plt.Rectangle(xy=(x1_a, y1_a), width = 5, height = 10, angle=ang_aa, color='black',ec='black',zorder=11)                  
               
                #后车轮
                square6 = plt.Rectangle(xy=(x1_a, y1_a), width = 5, height = -10, angle=ang_aa, color='black',ec='black',zorder=11)                  

        
                
                #画2个矩形用来遮挡车身上的黑色竖线 

                square7 = plt.Rectangle(xy=(x2_a, y2_a), width = 27, height = 0.2, angle=ang_a, color='royalblue',ec='royalblue',zorder=13)                  
                              
                square8 = plt.Rectangle(xy=(x2_a, y2_a), width = 27, height = -0.2, angle=ang_a, color='royalblue',ec='royalblue',zorder=13)                  
 


                #画车的血条

                if p1_a[i]==-1:
                    w_a += p1_a[i]*(25/step)
                else:
                    w_a -= 0
                    
            
                   
                           
                ax.add_patch(square1)
                ax.add_patch(square2)
                ax.add_patch(square3)
                ax.add_patch(square4)
                ax.add_patch(square5)
                ax.add_patch(square6)
                ax.add_patch(square6)
                ax.add_patch(square7)
                ax.add_patch(square8)
                

               

               

                
                
                
              

             
                x1_b = x_b[i]
                y1_b = y_b[i] 
                
                x2_b = x_b[i+1]
                y2_b = y_b[i+1]
               
                
                ang_b = angle_of_vector([x2_b-x1_b,y2_b-y1_b],[1,0])
    
                
                if x2_b-x1_b<0 and y2_b-y1_b<=0:
                    x1_b = x2_b - 27* math.cos(ang_b/180*np.pi)
                    y1_b = y2_b + 27* math.sin(ang_b/180*np.pi)
                    
                    ang_bb = -ang_b
                    ang_b = 180-ang_b 
                    
                  
                    
                    
                    x2_bb = x1_b + 40* math.cos(ang_b/180*np.pi)
                    y2_bb = y1_b - 40* math.sin(ang_b/180*np.pi)
                    ax.arrow(x1_b,y1_b,-x2_bb+x1_b,y2_bb-y1_b,length_includes_head = True,head_width = 13,head_length = 13,fc='red',ec='black',zorder=6)
         

                    
                elif x2_b-x1_b<=0 and y2_b-y1_b>0:
                    x1_b = x2_b - 27* math.cos(ang_b/180*np.pi)
                    y1_b = y2_b - 27* math.sin(ang_b/180*np.pi)
                                        
                    
                    ang_bb = ang_b
                    ang_b = ang_b-180 
                    
                    x2_bb = x1_b + 40* math.cos(ang_b/180*np.pi)
                    y2_bb = y1_b + 40* math.sin(ang_b/180*np.pi)          
                    ax.arrow(x1_b,y1_b,-x2_bb+x1_b,-y2_bb+y1_b,length_includes_head = True,head_width = 13,head_length = 13,fc='red',ec='black',zorder=6)
                   
    
                    
                elif x2_b-x1_b>0 and y2_b-y1_b>=0:
                    x1_b = x2_b - 27* math.cos(ang_b/180*np.pi)
                    y1_b = y2_b - 27* math.sin(ang_b/180*np.pi)                   
                    
                    ang_bb = ang_b
                    ang_b = ang_b-180 
                    
                    
                    x2_bb = x1_b + 40* math.cos(ang_b/180*np.pi)
                    y2_bb = y1_b + 40* math.sin(ang_b/180*np.pi)          
                    ax.arrow(x1_b,y1_b,-x2_bb+x1_b,-y2_bb+y1_b,length_includes_head = True,head_width = 13,head_length = 13,fc='red',ec='black',zorder=6)
                        
                    
    
                    
                elif x2_b-x1_b>=0 and y2_b-y1_b<0:
                    x1_b = x2_b - 27* math.cos(ang_b/180*np.pi)
                    y1_b = y2_b + 27* math.sin(ang_b/180*np.pi)
                                        
                    
                    ang_bb = -ang_b
                    ang_b = 180-ang_b   

                    x2_bb = x1_b + 40* math.cos(ang_b/180*np.pi)
                    y2_bb = y1_b - 40* math.sin(ang_b/180*np.pi)          
                    # ax.arrow(x1_b,y1_b,-x2_bb+x1_b,y2_bb-y1_b,length_includes_head = True,head_width = 13,head_length = 13,fc='limegreen',ec='black',zorder=6)
                    ax.arrow(x1_b,y1_b,-x2_bb+x1_b,y2_bb-y1_b,length_includes_head = True,head_width = 13,head_length = 13,fc='red',ec='black',zorder=6)  
                    

                

    
                
                                    
    
                #车身                           
                square9 = plt.Rectangle(xy=(x2_b, y2_b), width = 27, height = 7, angle=ang_b, color='red',ec='black',zorder=8)                  
               
                #车身
                square10 = plt.Rectangle(xy=(x2_b, y2_b), width = 27, height = -7, angle=ang_b, color='red',ec='black',zorder=8)                  
    
                #前车轮                          
                square11 = plt.Rectangle(xy=(x2_b, y2_b), width = 5, height = 10, angle=ang_b, color='black',ec='black',zorder=7)                  
               
                #前车轮
                square12 = plt.Rectangle(xy=(x2_b, y2_b), width = 5, height = -10, angle=ang_b, color='black',ec='black',zorder=7)                  
     
                #后车轮                          
                square13 = plt.Rectangle(xy=(x1_b, y1_b), width = 5, height = 10, angle=ang_bb, color='black',ec='black',zorder=7)                  
               
                #后车轮
                square14 = plt.Rectangle(xy=(x1_b, y1_b), width = 5, height = -10, angle=ang_bb, color='black',ec='black',zorder=7)                  
               


                #画2个矩形用来遮挡车身上的黑色竖线 

                square15 = plt.Rectangle(xy=(x2_b, y2_b), width = 27, height = 0.2, angle=ang_b, color='red',ec='red',zorder=9)                  
                              
                square16 = plt.Rectangle(xy=(x2_b, y2_b), width = 27, height = -0.2, angle=ang_b, color='red',ec='red',zorder=9)                  

        


                #画车的血条
                
                if p1_b[i]==-1:
                    w_b += p1_b[i]*(25/step)
                else:
                    w_b -= 0
                    
               
        

                                    
                ax.add_patch(square9)
                ax.add_patch(square10)
                ax.add_patch(square11)
                ax.add_patch(square12)
                ax.add_patch(square13)
                ax.add_patch(square14)
                ax.add_patch(square15)
                ax.add_patch(square16)
               
                
                
                
                
                # print(radi)
                if k!=len(O_x)-1:
                    o_x=O_x[k]
                    o_y=O_y[k]
                # print(o_y)
                for j in range(0,len(o_x)):
                    
                      x_r, y_r = (o_x[j],o_y[j])
                    
                      cir1 = Circle(xy = (x_r, y_r), radius=env.get_r(), facecolor= 'black', zorder=j+20)
                      ax.add_patch(cir1)
    
                      plt.axis('scaled')
    
                          # ax.axis('equal')   #加上会强制转换图为正方形图，后续调整坐标轴不好调
                k+=1
                


                #标注两小车起始点
                cir_a = Circle(xy = (x_a[0], y_a[0]), radius=5, facecolor= 'royalblue',ec='royalblue',zorder=2.5)
                ax.add_patch(cir_a)
    
                cir_b = Circle(xy = (x_b[0], y_b[0]), radius=5, facecolor= 'red',ec='red',zorder=2.5)
                ax.add_patch(cir_b)                  
                   
                # plt.xlim(x_lim_min-50,x_lim_max+50)
                # plt.ylim(y_lim_min-50,y_lim_max+50)     

                plt.rcParams['font.sans-serif'] = ['Times New Roman']
                # plt.title('Trajectory of Cars',fontfamily='Times New Roman',fontweight='heavy',fontsize=17)
                
                plt.scatter(x_a[0], y_a[0],color='royalblue',label='Start of Car A',zorder=0)
                # plt.plot(x_a, y_a, color='royalblue', label='Car A',linewidth=1.5,zorder=1)
                plt.plot(x_a[:i+1], y_a[:i+1], color='royalblue', label='Trajectory of Car A',linewidth=1.5,zorder=2.1)                
                
                plt.scatter(x_b[0], y_b[0],color='red',label='Start of Car B',zorder=1.5)            
                # plt.plot(x_b, y_b, color='red', label='Car B',linewidth=1.5, zorder=2)  
                plt.plot(x_b[:i+1], y_b[:i+1], color='red', label='Trajectory of Car B',linewidth=1.5, zorder=2)                  
    
                plt.scatter(o_x[0], o_y[0],color='black',label='Obstacle')
                



                
                # plt.legend(loc='upper right',prop={'family' : 'Times New Roman','weight':'heavy'}) # 标签位置
                plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0,prop={'family' : 'Times New Roman','weight':'heavy'}) # 标签位置
                    
                plt.xlabel('x(m)',fontfamily='Times New Roman',fontweight='heavy',fontsize=12)
                plt.ylabel('y(m)',fontfamily='Times New Roman',fontweight='heavy',fontsize=12)
                plt.yticks(fontproperties = 'Times New Roman', weight='heavy',size = 12)
                plt.xticks(fontproperties = 'Times New Roman', weight='heavy',size = 12)
                plt.grid(zorder=-10) 
                
                left=min(min(min(x_a),min(x_b)),min(min(y_a),min(y_b)))-80
                right=max(max(max(x_a),max(x_b)),max(max(y_a),max(y_b)))+80
                plt.xlim(left,right)
                plt.ylim(left,right)               


                    
                #对局结束的胜负说明
                mid = (plt.xlim()[1]+plt.xlim()[0])/2
                mid_left = (mid + plt.xlim()[0])/2
                mid_right = (plt.xlim()[1] + mid)/2
                
                square_a = plt.Rectangle(xy=(mid_left+w/2, plt.ylim()[1]-10), width = -w, height = 10, angle=0, color='white',ec='black',zorder=1000)                  
                square__a = plt.Rectangle(xy=(mid_left+w/2, plt.ylim()[1]-10), width = -w_a, height = 10, angle=0, color='royalblue',ec='black',zorder=1000)                                 
                square_b = plt.Rectangle(xy=(mid_right-w/2, plt.ylim()[1]-10), width = w, height = 10, angle=0, color='white',ec='black',zorder=1000)                  
                square__b = plt.Rectangle(xy=(mid_right-w/2, plt.ylim()[1]-10), width = w_b, height = 10, angle=0, color='red',ec='black',zorder=1000)  
                
                ax.add_patch(square_a)
                ax.add_patch(square__a)  
                ax.add_patch(square_b)
                ax.add_patch(square__b)  
  
                plt.text(mid_left, plt.ylim()[1]+20, 'Car A : '+str(round(w_a,1)), color='royalblue',ha='center',va='center',weight='heavy',fontsize=12,zorder=1000)
                plt.text((plt.xlim()[1]+plt.xlim()[0])/2, plt.ylim()[1]+20, ':', color='black',ha='center',va='center',weight='heavy',fontsize=12,zorder=1000)
                plt.text(mid_right, plt.ylim()[1]+20, 'Car B : '+str(round(w_b,1)), color='red',ha='center',va='center',weight='heavy',fontsize=12,zorder=1000)
                                                
                # if i==len(x_a)-2:
                if i>len(x_a)-6:
                    if w_a>w_b:
                        plt.text((plt.xlim()[1]+plt.xlim()[0])/2, (plt.ylim()[1]+plt.ylim()[0])/2, 'Car A Win!', color='deeppink',ha='center',va='center',weight='heavy',fontsize=50,zorder=1000)
                    elif w_a<w_b:
                        plt.text((plt.xlim()[1]+plt.xlim()[0])/2, (plt.ylim()[1]+plt.ylim()[0])/2, 'Car B Win!', color='deeppink',ha='center',va='center',weight='heavy',fontsize=50,zorder=1000)
                    elif w_a==w_b:
                        plt.text((plt.xlim()[1]+plt.xlim()[0])/2, (plt.ylim()[1]+plt.ylim()[0])/2, 'Draw Game!', color='deeppink',ha='center',va='center',weight='heavy',fontsize=50,zorder=1000)
               
                                   
                
                
                
                
                # #dpi较低                
                plt.savefig('./graph/paper/Good Result/test_result_'+str(j_)+'/Gif/Trajectory%d.png'%i,bbox_inches = 'tight',dpi=300)

                # #dpi较高
                # plt.savefig('./graph/paper/Good Result/test_result_'+str(j_)+'/Gif/Trajectory%d.png'%i,bbox_inches = 'tight',dpi=600)

                
            # # #合成GIF 
            # fig_ = []
            # for i in range(0, len(x_a)-1):
            #     # fig.append(imageio.imread("Trajectory" + str(i) + ".png"))
            #     fig_.append(imageio.imread("./graph/paper/Good Result/test_result_"+str(j_)+"/Gif/Trajectory" + str(i) + ".png"))    
            #     # fig.append(imageio.imread("./graph/paper/Good Result/test_result_21/Gif/Trajectory" + str(i) + ".png"))    
              
            # gif_1 = "./graph/paper/Good Result/test_result_"+str(j_)+"/Gif/Trajectory.gif"
            # imageio.mimsave(gif_1, fig_, duration = 0.001)

        env.close()

    
Episode = 3
step = 400
    
    
torch.set_num_threads(config['num_threads'])
os.environ['MKL_NUM_THREADS'] = str(config['num_threads'])
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
if not config['random_seed']:
    torch.cuda.manual_seed_all(config['seed'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
        
    
# method
env = CarCombatEnv()
sac = SAC(env)
sac.load_nets('./',29999)

run(env,sac)