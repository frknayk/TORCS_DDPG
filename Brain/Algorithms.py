# Essentials
import numpy as np
import random
import collections
import csv

# Essentials for Neural Network
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.autograd import Variable

# Utility Functions
from Utilities.Utils import ReplayBuffer,OU,gear_lookUp

# Neural Network Architecture
from Brain.NeuralArch import ActorNetwork,CriticNetwork

"""
Any RL algorithm must go here
"""
### DDPG part is taken from  "https://github.com/jastfkjg/DDPG_Torcs_PyTorch"
### and put it in a class
class DDPG():
    def __init__(self,HYPERPARAMS,paths):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.BATCH_SIZE             = HYPERPARAMS['batch_size']
        self.GAMMA                  = HYPERPARAMS['gamma']
        self.TAU                    = HYPERPARAMS['tau']
        self.epsilon                = HYPERPARAMS['epsilon']
        self.epsilon_min            = HYPERPARAMS['epsilon_min']
        self.EXPLORE                = HYPERPARAMS["ou_explore_rate"]

        self.HIDDEN1_UNITS_actor    = HYPERPARAMS['HIDDEN1_UNITS_actor']
        self.HIDDEN2_UNITS_actor    = HYPERPARAMS['HIDDEN2_UNITS_actor']
        self.HIDDEN1_UNITS_critic   = HYPERPARAMS['HIDDEN1_UNITS_critic']
        self.HIDDEN2_UNITS_critic   = HYPERPARAMS['HIDDEN2_UNITS_critic']
        self.LRA                    = HYPERPARAMS['policy_lr']
        self.LRC                    = HYPERPARAMS['critic_lr']
        self.state_size             = HYPERPARAMS['state_dim']
        self.action_size            = HYPERPARAMS['action_dim']

        self.OU_steer_mu            = HYPERPARAMS['OU_steer_mu']        
        self.OU_steer_theta         = HYPERPARAMS['OU_steer_theta']                
        self.OU_steer_sigma         = HYPERPARAMS['OU_steer_sigma']  
        self.OU_throttle_mu         = HYPERPARAMS['OU_throttle_mu']                        
        self.OU_throttle_theta      = HYPERPARAMS['OU_throttle_theta']                
        self.OU_throttle_sigma      = HYPERPARAMS['OU_throttle_sigma']
        self.OU_brake_mu            = HYPERPARAMS['OU_brake_mu']                        
        self.OU_brake_theta         = HYPERPARAMS['OU_brake_theta']                
        self.OU_brake_sigma         = HYPERPARAMS['OU_brake_sigma']
        self.OU_brake_stoch_mu      = HYPERPARAMS['OU_brake_stoch_mu']                        
        self.OU_brake_stoch_theta   = HYPERPARAMS['OU_brake_stoch_theta']                
        self.OU_brake_stoch_sigma   = HYPERPARAMS['OU_brake_stoch_sigma']

        # Init Ornstein-Uhlenbeck Noise
        self.OU = OU()
        self.noise_t = np.zeros([1, 3])

        # Change Hidden Layer Dimensions From Directly the Definiton
        self.actor = ActorNetwork(self.state_size, self.HIDDEN1_UNITS_actor,self.HIDDEN2_UNITS_actor).to(self.device)
        self.actor.apply(self.init_weights)
        self.critic = CriticNetwork(self.state_size, self.action_size,self.HIDDEN1_UNITS_critic,self.HIDDEN2_UNITS_critic).to(self.device)
 
        # Loading Weights
        print("Loading Trained Actor Weights ")
        try:
            self.actor.load_state_dict(torch.load( paths['actor_path_choosen'] ))
            self.actor.eval()
            print("Actor model loaded successfully")
        except:
            print("cannot find the Actor model")

        print("Loading Trained Critic Weights ")   
        try:
            self.critic.load_state_dict(torch.load( paths['critic_path_choosen'] ))
            self.critic.eval()
            print("Critic model loaded successfully")
        except:
            print("cannot find the Critic model")

        self.buff                   = ReplayBuffer(HYPERPARAMS['buffer_size'])
        # print("Loading Memory ")   
        # try:
        #     self.critic.load_state_dict(torch.load( saved_weights['critic_path_choosen'] ))
        #     self.critic.eval()
        #     print("Memory loaded successfully")
        # except:
        #     print("cannot find the Memory")


        self.speed_X                = 0
        self.s_t  = 0
        self.s_t1 = 0
        self.action = 0       

        self.target_actor = ActorNetwork(self.state_size, self.HIDDEN1_UNITS_actor,self.HIDDEN2_UNITS_actor).to(self.device)
        self.target_critic = CriticNetwork(self.state_size, self.action_size,self.HIDDEN1_UNITS_critic,self.HIDDEN2_UNITS_critic).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_actor.eval()
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_critic.eval()

        self.criterion_critic = torch.nn.MSELoss(reduction='sum')
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.LRA)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.LRC)
        
        if torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor') 

    # Deciding in Train
    def decide(self,s_t,ob,if_train = True):
        self.s_t = s_t
        loss = 0
        self.epsilon -= 1.0 / self.EXPLORE
        a_t = np.zeros([1, self.action_size])
        noise_t = np.zeros([1, self.action_size])

        a_t_original = self.actor(torch.tensor(s_t.reshape(1, s_t.shape[0]), device=self.device).float())

        if torch.cuda.is_available():
            a_t_original = a_t_original.data.cpu().numpy()
        else:
            a_t_original = a_t_original.data.numpy()

        if if_train == True :
            noise_t[0][0] = max(self.epsilon, 0) * self.OU.function(a_t_original[0][0], self.OU_steer_mu,    self.OU_steer_theta    , self.OU_steer_sigma)
            noise_t[0][1] = max(self.epsilon, 0) * self.OU.function(a_t_original[0][1], self.OU_throttle_mu, self.OU_throttle_theta , self.OU_throttle_sigma)
            noise_t[0][2] = max(self.epsilon, 0) * self.OU.function(a_t_original[0][2], self.OU_brake_mu,    self.OU_brake_theta    , self.OU_brake_sigma)

            # Stochastic brake
            if random.random() <= 0.1:
                print("Applying the brake")
                noise_t[0][2] = max(self.epsilon, 0) * self.OU.function(a_t_original[0][2], self.OU_brake_stoch_mu, self.OU_brake_stoch_theta, self.OU_brake_stoch_sigma)

            a_t[0][0] = a_t_original[0][0] + noise_t[0][0]
            a_t[0][1] = a_t_original[0][1] + noise_t[0][1]
            a_t[0][2] = a_t_original[0][2] + noise_t[0][2]
        else : 
            a_t = a_t_original
            speed_X = self.speed_X
            # Brake Signal is too smol
            a_t[0][2] = a_t[0][2] * 1
            a_t[0][1] = a_t[0][1] * 0.70
            
            print("Throttle(notSat) : {0}".format(a_t[0][1]) )
            print("Brake(notSat)    : {0}".format(a_t[0][2]) )

            a_t[0][0] = a_t[0][0] * 0.50

            if a_t[0][0] > 0.261799388 :
                a_t[0][0] =  0.261799388
            elif a_t[0][0] < -0.261799388 :
                a_t[0][0] =  -0.261799388

            if a_t[0][1] > a_t[0][2] :
                a_t[0][2] =  0.0
            elif a_t[0][1] < a_t[0][2] and speed_X > 25:
                a_t[0][1] = 0.0  

            if speed_X > 5:
                a_t[0][2] = 0

            # Brake Final Sat
            if a_t[0][2] > 1 :
                a_t[0][2] =  1.0
            elif a_t[0][2] < 0 :
                a_t[0][2] =  0
            
            print("Steer            : {0}".format(a_t[0][0]) )
            print("Throttle(sat)    : {0}".format(a_t[0][1]) )
            print("Brake(sat)       : {0}".format(a_t[0][2]) )
            print("Longitudunal Vx  : {0}".format(speed_X) )
            print("*************************************")


        return a_t

    def train(self,s_t1,s_t,ob,a_t,r_t,done):

        #add to replay buffer
        self.buff.add(s_t, a_t[0], r_t, s_t1, done)

        batch = self.buff.getBatch(self.BATCH_SIZE)

        states      = torch.tensor(np.asarray([e[0] for e in batch]), device=self.device).float()    #torch.cat(batch[0])
        actions     = torch.tensor(np.asarray([e[1] for e in batch]), device=self.device).float()
        rewards     = torch.tensor(np.asarray([e[2] for e in batch]), device=self.device).float()
        new_states  = torch.tensor(np.asarray([e[3] for e in batch]), device=self.device).float()
        dones       = np.asarray([e[4] for e in batch])
        y_t         = torch.tensor(np.asarray([e[1] for e in batch]), device=self.device).float()
        
        #use target network to calculate target_q_value
        target_q_values = self.target_critic(new_states, self.target_actor(new_states))

        for k in range(len(batch)):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + self.GAMMA * target_q_values[k]
        
        # Training
        q_values    = self.critic(states, actions)
        loss        = self.criterion_critic(y_t, q_values)  
        self.optimizer_critic.zero_grad()
        loss.backward(retain_graph=True)                         ##for param in critic.parameters(): param.grad.data.clamp(-1, 1)
        self.optimizer_critic.step()
        a_for_grad = self.actor(states)
        a_for_grad.requires_grad_()    #enables the requires_grad of a_for_grad
        q_values_for_grad = self.critic(states, a_for_grad)
        self.critic.zero_grad()
        q_sum = q_values_for_grad.sum()
        q_sum.backward(retain_graph=True)
        grads = torch.autograd.grad(q_sum, a_for_grad) #a_for_grad is not a leaf node  

        act = self.actor(states)
        self.actor.zero_grad()
        act.backward(-grads[0])
        self.optimizer_actor.step()

        # soft update for target network
        new_actor_state_dict = collections.OrderedDict()
        new_critic_state_dict = collections.OrderedDict()
        for var_name in self.target_actor.state_dict():
            new_actor_state_dict[var_name] = self.TAU * self.actor.state_dict()[var_name] + (1-self.TAU) * self.target_actor.state_dict()[var_name]
        self.target_actor.load_state_dict(new_actor_state_dict)
        for var_name in self.target_critic.state_dict():
            new_critic_state_dict[var_name] = self.TAU * self.critic.state_dict()[var_name] + (1-self.TAU) * self.target_critic.state_dict()[var_name]
        self.target_critic.load_state_dict(new_critic_state_dict)

    def save_model(self,paths,eps,reward):
        
        print("Saving model and replay buffer")
        
        str_t_p     =   paths['policy_save_path'] 
        str_t_c     =   paths['critic_save_path']
        str_buff    =   paths['RepBuff_save_path']
        str_reward  =   paths['reward_save_path']

        str_t_p  = str_t_p  + str(eps)
        str_t_c  = str_t_c  + str(eps)
        str_buff = str_buff + str(eps)

        # Save Actor Weights
        torch.save(self.actor.state_dict(), str_t_p)

        # Save Critic Weights
        torch.save(self.critic.state_dict(), str_t_c)

        # Save Replay Buffer
        np.save(str_buff,self.buff.buffer)

        # Save Reward to CSV ( for plotting later )
        with open(str_reward, 'a') as csv_file:
            str_rew = str(reward) + "\n"
            csv_file.write( str_rew )
        
    # Zeroes the injected Ornstein-Uhlenbeck noise
    def reset(self):
        self.noise_t = np.zeros([1, 3])

    def init_weights(self,m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.normal_(m.weight, 0, 1e-4)
            m.bias.data.fill_(0.0)
