import numpy as np
import math

""" 
All of the network parameters are below
Just change any parameters of the network and DDPG here, 
so you don't need to change anything inside the main code (train.py)
"""
HYPERPARAMS = {

        ### Parameters about Policy and Critic Networks
        'policy_lr'             : 0.0001,
        'critic_lr'             : 0.001,
        
        'state_dim'             : 29,   
        'action_dim'            : 3,    

        'HIDDEN1_UNITS_actor'   : 300,
        'HIDDEN2_UNITS_actor'   : 600,
        
        'HIDDEN1_UNITS_critic'  : 300,
        'HIDDEN2_UNITS_critic'  : 600,

        'buffer_size'           : 100000,
        'batch_size'            : 32,
        'gamma'                 : 0.99,
        'tau'                   : 0.001, 
        'ou_explore_rate'       : 100000. ,

        ### Parameters About Training Loop
        'epoch'                 : 2000,
        'epsilon'               : 1,
        'epsilon_min'           : 0.07,
        'max_steps'             : 100000,

        ### OU Noise Parameters for Steer,Throttle and Brakse
        #OU_noise = theta * (mu - x) + sigma * np.random.randn(1)
        'OU_steer_mu'           : 0.0,        
        'OU_steer_theta'        : 0.60, 
        'OU_steer_sigma'        : 0.30,   

        'OU_throttle_mu'        :  0.5  ,                        
        'OU_throttle_theta'     :  1.00 ,                
        'OU_throttle_sigma'     :  0.10 ,

        'OU_brake_mu'           : -0.1 ,        
        'OU_brake_theta'        : 1.00 ,                
        'OU_brake_sigma'        : 0.05 ,

        'OU_brake_stoch_mu'     : 0.2 ,        
        'OU_brake_stoch_theta'  : 1.00 ,                
        'OU_brake_stoch_sigma'  : 0.10 ,  


        }