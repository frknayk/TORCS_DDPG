import torch
from torch.autograd import Variable
import numpy as np
import random
from gym_torcs import TorcsEnv
import argparse
import collections
#import ipdb

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU

state_size = 29
action_size = 3
LRA = 0.0001
LRC = 0.001
BUFFER_SIZE = 100000  #to change
BATCH_SIZE = 32
GAMMA = 0.95
EXPLORE = 100000.
epsilon = 1
train_indicator = 0    # train or not
TAU = 0.001

VISION = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OU = OU()

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 1e-4)
        m.bias.data.fill_(0.0)


actor = ActorNetwork(state_size).to(device)
actor.apply(init_weights)
critic = CriticNetwork(state_size, action_size).to(device)

"""
Tracks that agent accomplished succesfully,
1. Speedway
2. E-Road
3. CG-track 2
4. CG-track 3 ( Most of it )
"""
#load model
print("loading actor network")
try:
    # actor.load_state_dict(torch.load('actormodel.pth'))
    # actor.load_state_dict(torch.load('actor/actormodel_best.pth'))
    actor.load_state_dict(torch.load('Best Weights/actormodel_best_SW_Forza_416K.pth'))
    # actor.load_state_dict(torch.load('actor/actormodel_43.pth'))
    actor.eval()
    print("model load successfully")
except:
    print("cannot find actor the model")

print("loading critic network")
try:
    critic.load_state_dict(torch.load('criticmodel.pth'))
    # critic.load_state_dict(torch.load('critic/criticmodel_best.pth'))
    critic.eval()
    print("model load successfully")
except:
    print("cannot find critic the model")

#critic.apply(init_weights)
buff = ReplayBuffer(BUFFER_SIZE)

target_actor = ActorNetwork(state_size).to(device)
target_critic = CriticNetwork(state_size, action_size).to(device)
target_actor.load_state_dict(actor.state_dict())
target_actor.eval()
target_critic.load_state_dict(critic.state_dict())
target_critic.eval()

criterion_critic = torch.nn.MSELoss(reduction='sum')

optimizer_actor = torch.optim.Adam(actor.parameters(), lr=LRA)
optimizer_critic = torch.optim.Adam(critic.parameters(), lr=LRC)

#env environment
env = TorcsEnv(vision=VISION, throttle=True, gear_change=False,if_train=False)

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor') 

if(train_indicator == 1):
    print("***************************************************")
    print("Be Cautios !!!!\nTraining Will Be Performed !")
    print("***************************************************")



speed_X = 0 
for i in range(2000):

    if np.mod(i, 3) == 0:
        ob = env.reset(relaunch = True)
    else:
        ob = env.reset()

    s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))
    
    speed_X = 0

    for j in range(100000):
        loss = 0
        epsilon -= 1.0 / EXPLORE
        a_t = np.zeros([1, action_size])
        noise_t = np.zeros([1, action_size])
        #ipdb.set_trace() 
        a_t_original = actor(torch.tensor(s_t.reshape(1, s_t.shape[0]), device=device).float())

        if torch.cuda.is_available():
            a_t_original = a_t_original.data.cpu().numpy()
        else:
            a_t_original = a_t_original.data.numpy()
        
        #print(type(a_t_original[0][0]))

        noise_t[0][0] = 0
        noise_t[0][1] = 0
        noise_t[0][2] = 0

        a_t = a_t_original

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

        ob, r_t, done, info = env.step(a_t[0])
        speed_X = ob.speedX * 300

        s_t1 = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

        s_t = s_t1


        if done:
            print("---Episode ", i , "  Action:", a_t, "  Reward:", r_t)
            break
    
env.end()
print("Finish.")

#for param in critic.parameters(): param.grad.data.clamp(-1, 1)