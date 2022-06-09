# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 09:32:47 2021

@author: 86153
"""

import math
import numpy as np
import random
import gym
from gym import spaces
from gym.utils import seeding
from math import sqrt, pow



class CarCombatEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, goal_velocity=0):

        self.min_position = -2000
        self.max_position = 2000
        self.min_theta = 0
        self.max_theta = 2*np.pi

        self.min_d = 0.1

 
        
        self.min_delta = -np.pi/4
        self.max_delta = np.pi/4
        self.min_velocity = 2
        self.max_velocity = 50
        self.min_a = -10
        self.max_a = 10
        
        self.dt = 0.1
        self.l_r = 2.5
        self.l_f = 2.5
        

        self.low_state = np.array(
            [self.min_position, self.min_position, self.min_theta, self.min_velocity, self.min_position, self.min_position, self.min_theta, self.min_velocity, -float('inf'),-float('inf')], dtype=np.float32
        )
        self.high_state = np.array(
            [self.max_position, self.max_position, self.max_theta, self.max_velocity, self.max_position, self.max_position, self.max_theta, self.max_velocity, float('inf'), float('inf')], dtype=np.float32
        )

        self.low_action = np.array(
            [self.min_a, self.min_delta], dtype=np.float32
        )
        self.high_action = np.array(
            [self.max_a, self.max_delta], dtype=np.float32
        )        
        
        
        self.viewer = None

        self.action_space = spaces.Box(
            low=self.low_action,
            high=self.high_action,
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )

        self.seed()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # def step(self, action, action_2, episode,i,o_x,o_y):
    def step(self, action, action_2, episode,i):
            
        dt = self.dt   
                  
            
        # a_b = random.uniform(-10,10)
        # delta_b = random.uniform(-np.pi/4,np.pi/4)
        a_b = random.uniform(-10,10)
        delta_b = random.uniform(-np.pi/10,np.pi/10)
        
        a_b = random.uniform(-10,10)
        delta_b = 0
        
        
        # a_b = 1
        # delta_b = 0

        # if i<=200:
        #     a_b = 1
        #     delta_b = np.pi/100           
        # elif i>200 and i<=500:
        #     a_b = 1
        #     delta_b = np.pi/150         
        # elif i>500 and i<=800:
        #     a_b = 1
        #     delta_b = np.pi/200 
        # elif i>800 and i<=1000:
        #     a_b = 1
        #     delta_b = np.pi/250  
            
        # a_b = random.uniform(-5,5)
        # delta_b = random.uniform(-np.pi/4,np.pi/4)
        # a_b = action_2[0]
        # delta_b = action_2[1] 
        
        
        # if i<=200:
        #     a_b = 0
        #     delta_b = np.pi/100           
        # elif i>200 and i <=400:
        #     a_b = 0
        #     delta_b = -np.pi/100 
        # elif i>400:
        #     a_b = random.uniform(-10,10)
        #     delta_b = random.uniform(-np.pi/4,np.pi/4)


        # if i<=100:
        #     a_b = 2
        #     delta_b = np.pi/100           
        # elif i>100 and i<=200:
        #     a_b = 2
        #     delta_b = -np.pi/100 
        # elif i>200 and i<=300:
        #     a_b = 2
        #     delta_b = np.pi/100 
        # elif i>300 and i<=400:
        #     a_b = 2
        #     delta_b = -np.pi/100 
        # elif i>400 and i<=500:
        #     a_b = 2
        #     delta_b = np.pi/100 
        # elif i>500 and i<=600:
        #     a_b = 2
        #     delta_b = -np.pi/100 
        # elif i>600 and i<=700:
        #     a_b = 2
        #     delta_b = np.pi/100 
            
            
            
        # a_b = random.uniform(-10,10)
        # delta_b = random.uniform(-np.pi/4,np.pi/4)
        # a_b = 0.5
        # delta_b = 0
        
        # if i <200:
        #     a_b = random.uniform(-5,5)
        #     delta_b = random.uniform(-np.pi/4,np.pi/4)
        # else:           
        #     a_b = action_2[0]
        #     delta_b = action_2[1]




        # a_b = action_2[0]
        # delta_b = action_2[1]                

        

        # #方案一：渐进式训练     
        # global o_x
        # global o_y
        
        # o_x = []
        # o_y = []
        
        # global r
            

        # r=10
        # for k in range(0,15):          
        #     o_x.append(random.uniform(-200,0))
        #     o_y.append(random.uniform(-200,200))

        # def get_obstacle(self):
        #     return o_x, o_y


        
        global v_a
        global v_b
        
        
        
        x_a = self.state[0]
        y_a = self.state[1]
        theta_a = self.state[2]
        v_a = self.state[3]
        x_b = self.state[4]
        y_b = self.state[5]
        theta_b = self.state[6]
        v_b = self.state[7]
        
        a_a = min(max(action[0], self.min_a), self.max_a)
        delta_a = min(max(action[1], self.min_delta), self.max_delta)

        beta_a = math.atan((self.l_r / (self.l_r + self.l_f)) * math.tan(delta_a))
        x_a_ = v_a * math.cos(theta_a + beta_a)
        y_a_ = v_a * math.sin(theta_a + beta_a)
        theta_a_ = (v_a /self.l_f) * math.sin(beta_a)
        v_a_ = a_a

        beta_b = math.atan((self.l_r / (self.l_r + self.l_f)) * math.tan(delta_b))
        x_b_ = v_b * math.cos(theta_b + beta_b)
        y_b_ = v_b * math.sin(theta_b + beta_b)
        theta_b_ = (v_b /self.l_f) * math.sin(beta_b)
        v_b_ = a_b


        #我方小车状态更新      
        x_a = x_a + x_a_ * dt
        if (x_a >= self.max_position): x_a = self.max_position
        if (x_a <= self.min_position): x_a = self.min_position        
        
        y_a = y_a + y_a_ * dt
        if (y_a >= self.max_position): y_a = self.max_position
        if (y_a <= self.min_position): y_a = self.min_position      
        
        theta_a = theta_a + theta_a_ * dt
        if (theta_a > self.max_theta): theta_a = theta_a - theta_a//(2*np.pi)*2*np.pi
        if (theta_a < self.min_theta): theta_a = theta_a - (theta_a//(2*np.pi)-1)*2*np.pi       
        
        v_a = v_a + v_a_ * dt
        if (v_a >= self.max_velocity): v_a = self.max_velocity
        if (v_a <= self.min_velocity): v_a = self.min_velocity




        #对方小车状态更新  
        x_b = x_b + x_b_ * dt
        if (x_b >= self.max_position): x_b = self.max_position
        if (x_b <= self.min_position): x_b = self.min_position           
        
        y_b = y_b + y_b_ * dt
        if (y_b >= self.max_position): y_b = self.max_position
        if (y_b <= self.min_position): y_b = self.min_position             
        
        theta_b = theta_b + theta_b_ * dt
        if (theta_b > self.max_theta): theta_b = theta_b - theta_b//(2*np.pi)*2*np.pi
        if (theta_b < self.min_theta): theta_b = theta_b - (theta_b//(2*np.pi)-1)*2*np.pi       

        v_b = v_b + v_b_ * dt
        if (v_b >= self.max_velocity): v_b = self.max_velocity
        if (v_b <= self.min_velocity): v_b = self.min_velocity
        

  

 

        global d
        d = math.sqrt(((x_a + self.l_f * math.cos(theta_a)) - (x_b - self.l_r * math.cos(theta_b)))**2 + ((y_a + self.l_f * math.sin(theta_a)) - (y_b - self.l_r * math.sin(theta_b)))**2)
        
        # l_1 = np.array([self.l_f * math.cos(theta_a), self.l_f * math.sin(theta_a)])    #车身所在的向量    
        # l_2 = np.array([x_b-self.l_r*math.cos(theta_b)-x_a-self.l_f * math.cos(theta_a), y_b - self.l_r * math.sin(theta_b)-y_a - self.l_f * math.sin(theta_a)])   #对方车尾与我方车头之间的向量

        v_1 = [v_a * math.cos(theta_a + beta_a),v_a * math.sin(theta_a + beta_a)]  #我方速度方向所在的向量 
        v_2 = [v_b * math.cos(theta_b + beta_b),v_b * math.sin(theta_b + beta_b)]  #对方速度方向所在的向量 
        # p_a = [x_a,y_a]  #我方小车位置矢量
        # p_b = [x_b,y_b]  #对方小车位置矢量
        p_ab = [x_b-x_a,y_b-y_a]
        p_ba = [x_a-x_b,y_a-y_b] 
        
        l_1 = [self.l_f * math.cos(theta_a), self.l_f * math.sin(theta_a)]    #我方车身所在的向量    
        l_2 = [self.l_f * math.cos(theta_b), self.l_f * math.sin(theta_b)]    #对方车身所在的向量          
        l_3 = [x_b-self.l_r*math.cos(theta_b)-x_a-self.l_f * math.cos(theta_a), y_b - self.l_r * math.sin(theta_b)-y_a - self.l_f * math.sin(theta_a)]  #我方车头与对方车尾之间的向量        
        l_4 = [x_a-self.l_r*math.cos(theta_a)-x_b-self.l_f * math.cos(theta_b), y_a - self.l_r * math.sin(theta_a)-y_b - self.l_f * math.sin(theta_b)]  #对方车尾与我方车头之间的向量           
        global angle_1
        global angle_2        
        angle_1 = self.angle_of_vector(v_1, p_ab)   #我方和对方连线与我方车身方向的夹角
        angle_2 = self.angle_of_vector(v_2, p_ba)   #对方和我方连线与对方车身方向的夹角        


        #计算出小车运动方向上速度方向直对的障碍物的距离
        n=[]
        m=[]
        for k in range(len(o_x)):
            alpha = math.acos(self.cos_angle_of_vector(v_1,[o_x[k]-x_a,o_y[k]-y_a]))
            cos_alpha = self.cos_angle_of_vector(v_1,[o_x[k]-x_a,o_y[k]-y_a])
            d_o=math.sqrt((o_x[k]-x_a)**2+(o_y[k]-y_a)**2)
            m.append(d_o)
            if d_o>=r:
                
                if alpha <= math.acos(math.sqrt(d_o**2-r**2)/(d_o)):
                    d_min = d_o*cos_alpha-math.sqrt(d_o**2*(cos_alpha**2-1)+r**2)
                    n.append(d_min)
            else:
                n.append(0)
                    
        if n==[]:
            
            d_min = 1000
        else:
            # if min(m)==0 or min(m)-2<=0:
            #     d_min=0
            # elif min(m)-2>0:
            #     d_min=min(m)-2
                
            d_min= min(n)
            

            

                




        # global m    
        # m=[]
        # for j in range(0,len(o_x)):
        #     m.append(math.sqrt((x_a-o_x[j])**2 + (y_a-o_y[j])**2))

        # p_d = [o_x[m.index(min(m))],o_y[m.index(min(m))]]  #最近障碍物的空间位置向量
        

        
        if d <= 0.5 or min(m)<r:
                done=False
        else:
                done=False
                
        # def get_min_m():
        #     return min(m)

        reward = 0
        
        w1=2
        w2=2
        w3=2
        
        R_A = (w1 * self.cos_angle_of_vector(v_1,v_2) + w2 * self.cos_angle_of_vector(v_1,[x_b-x_a,y_b-y_a]))/(math.sqrt((x_a-x_b+12.5*math.cos(theta_b))**2 + (y_a - y_b +12.5*math.sin(theta_b))**2) + 0.0001)        
        R_1_A = -w3 * (v_a/(d_min + 0.00001))
        reward += 50*R_A + R_1_A       
        # reward += 50*R_A
        # reward += R_1_A
        if done:
            reward = -500
            
            
        reward_scaling=1

        reward = reward * reward_scaling

        # self.state = np.array([x_a, y_a, theta_a, v_a, x_b, y_b, theta_b, v_b])
        self.state = np.array([x_a, y_a, theta_a, v_a, x_b, y_b, theta_b, v_b, d, d_min, delta_b, a_b])
        
        # self.state_2 = np.array([x_b, y_b, theta_b, v_b, x_a, y_a, theta_a, v_a, d, min(m)])    
        
        
        # return self.state, reward, done, {}
        
        reward__= 50*(w1 * self.cos_angle_of_vector(l_1,l_2) + w2 * self.cos_angle_of_vector(l_1,[x_b-x_a,y_b-y_a]))/(math.sqrt((x_a-x_b+12.5*math.cos(theta_b))**2 + (y_a - y_b +12.5*math.sin(theta_b))**2) + 0.0001)        
          
    #   切记带上随机障碍物的位置
        # return self.state, reward, done, {}, o_x, o_y    
        return self.state, reward, done, {}, o_x, o_y,reward__    

    
    def get_distance(self):       
        return d
    def get_v_a(self):
        return v_a
    def get_v_b(self):
        return v_b
    
    def get_obstacle(self):
        return o_x, o_y
        

    def reset(self):
        
        
        #方案一：渐进式训练     
        global o_x
        global o_y
        
        o_x = []
        o_y = []
        
        global r
            

        r=0.1
        for k in range(0,2):          
            o_x.append(random.uniform(-300,300))
            o_y.append(random.uniform(-300,300))
        
        

        
        
        # o_x = [-400,-400,-400,-500,-500,-500,-600,-600,-600,-300,-300,-300,-250,-250,-250,-200,-200,-200,-150,-150,-150,-100,-100,-100,-50,-50,-50,0,0,0]
        # o_y = [100,0,-100]*10
        
        # o_x = [-1000,-1000,-1000,-1000,-1000,-900,-900,-900,-900,-900,-800,-800,-800,-800,-800,-700,-700,-700,-700,-700,-600,-600,-600,-600,-600,-500,-500,-500,-500,-500,-400,-400,-400,-400,-400,-300,-300,-300,-300,-300,-200,-200,-200,-200,-200,-100,-100,-100,-100,-100,0,0,0,0,0]
        # o_y = [100,0,-100,-200,-300]*11        


        # o_x = [-300,-300,-300,-300,-300,-200,-200,-200,-200,-200,-100,-100,-100,-100,-100,0,0,0,0,0]
        # o_y = [100,0,-100,-200,-300]*4
        # o_x = [-300,-300,-300,-200,-200,-200,-100,-100,-100,0,0,0]
        # o_y = [0,-100,-200]*4  
        
        
        
        # o_x = [100,-200,0,210,-280,250,-110,-5,0,-200,-380,-110]
        # o_y = [30,0,-100,100,100,-100,250,0,300,-200,-100,0]  

        # o_x = [0,0,0,0,0,0,0,0,0,0,0,0]
        # o_y = [-20,20,60,-60,100,-100, 140, -140,180,-180,220,-220]
        
        # o_x = [0,0,0,0,0,0]
        # o_y = [-20,20,60,-60,100,-100] 
        
        # o_x = [0,9*2**0.5,18*2**0.5,27*2**0.5,36*2**0.5,9*2**0.5,18*2**0.5,27*2**0.5,36*2**0.5]
        # o_y = [0,9*2**0.5,18*2**0.5,27*2**0.5,36*2**0.5,-9*2**0.5,-18*2**0.5,-27*2**0.5,-36*2**0.5]  

       
        # o_x = [0,9*2**0.5,18*2**0.5,27*2**0.5,36*2**0.5,45*2**0.5,9*2**0.5,18*2**0.5,27*2**0.5,36*2**0.5,45*2**0.5]
        # o_y = [0,9*2**0.5,18*2**0.5,27*2**0.5,36*2**0.5,45*2**0.5,-9*2**0.5,-18*2**0.5,-27*2**0.5,-36*2**0.5,-45*2**0.5]  
        
        
        # o_x = [0,0,0,0,0,0, -50,-50,-50,-50,-50,-50, -100,-100,-100,-100,-100,-100, -150,-150,-150,-150,-150,-150, -200,-200,-200,-200,-200,-200, ]
        # o_y = [-20,20,60,-60,100,-100, -70,-30,10,-110,50,-150, -20,20,60,-60,100,-100, -70,-30,10,-110,50,-150, -20,20,60,-60,100,-100, ]


        
        # o_x = [-50]
        # o_y = [50]
        
        # o_x = [-200]
        # o_y = [-100]  

              
        # #方案二：难度拉满的暴力训练     
        # global o_x
        # global o_y
        
        # o_x = []
        # o_y = []
        
        # global r            
        # random.seed(episode)             
        # r=random.uniform(8,15)
        
        # for k in range(0,20):
        #     o_x.append(random.uniform(-400,400))
        #     o_y.append(random.uniform(-400,400))  
                    
        
        
        
        # a= np.array([random.uniform(-10,10), random.uniform(-10,10),self.np_random.uniform(low=0, high=2*np.pi), 30, random.uniform(-10,10), random.uniform(-10,10), self.np_random.uniform(low=0, high=2*np.pi), 30])
        

         
        a= np.array([self.np_random.uniform(-45,45), self.np_random.uniform(-45,45),self.np_random.uniform(low=0, high=2*np.pi), self.np_random.uniform(10,30), self.np_random.uniform(-35,35), self.np_random.uniform(-35,35), self.np_random.uniform(low=0, high=2*np.pi), self.np_random.uniform(10,30)])
        a= np.array([-30, 0,0, 25, 30, 0, 0, 25])
        a= np.array([30, 0,np.pi/2, 25, -30, 0, np.pi, 25])     
        a= np.array([80, 0,np.pi/2, 25, -80, 0, np.pi, 1]) 
        # a= np.array([80, 0, np.pi/2, 25, -80, 0, np.pi, 10])   #用于对抗  
        # a= np.array([30, 0,np.pi/2, 25, -30, 0, np.pi, 25])  
        a= np.array([80, 0,np.pi/2, 25, -80, 0, np.pi, 25]) 
        a= np.array([20, 0, np.pi/2, 20, -20, 0, np.pi/2, 10])   #用于对抗 
        a= np.array([self.np_random.uniform(-10,10), self.np_random.uniform(-10,10),self.np_random.uniform(low=0, high=2*np.pi), 10, self.np_random.uniform(-10,10), self.np_random.uniform(-10,10), self.np_random.uniform(low=0, high=2*np.pi), 10])
        # a= np.array([-50, 0, np.pi/2, 20, 80, 0, np.pi/2, 20])   #用于对抗  
        
        
        delta_b_init = 0
        a_b_init = 0
        d = math.sqrt(((a[0] + self.l_f * math.cos(a[2])) - (a[4] - self.l_r * math.cos(a[6])))**2 + ((a[1] + self.l_f * math.sin(a[2])) - (a[5] - self.l_r * math.sin(a[6])))**2)
        self.state = np.array([a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],d,20, delta_b_init, a_b_init])
            
      
                
        return np.array(self.state)

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55
    
    def angle_of_vector(self,v1, v2):
        pi = np.pi
        vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
        length_prod = sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * sqrt(pow(v2[0], 2) + pow(v2[1], 2))
        cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
        return (math.acos(cos) / pi) * 180
    def cos_angle_of_vector(self,v1, v2):
        pi = np.pi
        vector_prod = v1[0] * v2[0] + v1[1] * v2[1]
        length_prod = sqrt(pow(v1[0], 2) + pow(v1[1], 2)) * sqrt(pow(v2[0], 2) + pow(v2[1], 2))
        cos = vector_prod * 1.0 / (length_prod * 1.0 + 1e-6)
        return cos



    def get_min_m(self,x_b,y_b,theta_b,v_b):
        o_x, o_y = self.get_obstacle()
     
        beta_b = math.atan((self.l_r / (self.l_r + self.l_f)) * math.tan(np.pi/6))        
        v_2 = [v_b * math.cos(theta_b + beta_b),v_b * math.sin(theta_b + beta_b)] 
            
        #计算出小车运动方向上速度方向直对的障碍物的距离
        n=[]
        for k in range(len(o_x)):
            alpha = math.acos(self.cos_angle_of_vector(v_2,[o_x[k]-x_b,o_y[k]-y_b]))
            cos_alpha = self.cos_angle_of_vector(v_2,[o_x[k]-x_b,o_y[k]-y_b])
            d_o=math.sqrt((o_x[k]-x_b)**2+(o_y[k]-y_b)**2)
            if d_o>=r:
                
                if alpha <= math.acos(math.sqrt(d_o**2-r**2)/(d_o)):
                    d_min = d_o*cos_alpha-math.sqrt(d_o**2*(cos_alpha**2-1)+r**2)
                    n.append(d_min)
                    
        if n==[]:
            
            d_min = 1000
        else:
            # d_min= max(min(n)-5,2)
            d_min= min(n)            
            
            

        return d_min
    def get_b_step_reward(self,x_a, y_a, theta_a, v_a, x_b, y_b, theta_b, v_b, o_x, o_y):
        reward=0
        m=[]
        for j in range(0,len(o_x)):
            m.append(math.sqrt((x_b-o_x[j])**2 + (y_b-o_y[j])**2))
    
        l_1 = [self.l_f * math.cos(theta_a), self.l_f * math.sin(theta_a)]    #我方车身所在的向量    
        l_2 = [self.l_f * math.cos(theta_b), self.l_f * math.sin(theta_b)]    #对方车身所在的向量          
        l_3 = [x_b-self.l_r*math.cos(theta_b)-x_a-self.l_f * math.cos(theta_a), y_b - self.l_r * math.sin(theta_b)-y_a - self.l_f * math.sin(theta_a)]  #我方车头与对方车尾之间的向量        
        l_4 = [x_a-self.l_r*math.cos(theta_a)-x_b-self.l_f * math.cos(theta_b), y_a - self.l_r * math.sin(theta_a)-y_b - self.l_f * math.sin(theta_b)]  #对方车尾与我方车头之间的向量           


        d = self.get_distance()     
       
        if d <= 0.5 or min(m)<=r:
                done_b=False
        else:
                done_b=False
                
        # def get_min_m():
        #     return min(m)

        reward = 0
        w1=2
        w2=2

        
        R_B = (w1 * self.cos_angle_of_vector(l_2,l_1) + w2 * self.cos_angle_of_vector(l_2,[x_a-x_b,y_a-y_b]))/(math.sqrt((x_b-x_a+12.5*math.cos(theta_a))**2 + (y_b - y_a +12.5*math.sin(theta_a))**2) + 0.0001)        
  
               
        reward = 50*R_B
        return reward
    def get_angle(self):
        return angle_1, angle_2     
    def get_r(self):
        return r