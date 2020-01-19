import math
import numpy as np
import random
from collections import deque

# Ornstein-Uhlenbeck Noise
class OU():
    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)

class ReplayBuffer():
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0

# def gear(state):
#       gearup   =[7000 , 7000 , 7000 , 7000 , 7000 , 0]
#       geardown =[0 , 2500 , 3000 , 3000 , 3500 , 3500]
#       gear = state.getGear()
#       rpm = state.getRpm()
#       if gear < 1 :
#           gear = 1 
#       if(gear < 6) and (rpm  >= gearup[gear -1]):
#           gear = gear +1 ;
#       elif(gear > 1) and (rpm <= geardown[gear -1]):
#           gear =  gear -1
#       else:
#           gear = gear
#       control.setGear(gear)

# Look up table based gear Logic
def gear_lookUp(state):
    gear = state.gear
    engine_Rpm = state.rpm*10000
    
    ### GEAR 0
    if ( int(state.gear) == 0):
        gear = 1
    
    ### GEAR 1
    if ( int(state.gear) == 1):
        if( engine_Rpm > 5500):
            gear = 2

    ### GEAR 2
    if ( int(state.gear) == 2):
        if( engine_Rpm < 1500):
            gear = 1
        elif( engine_Rpm > 6000):
            gear = 3

    ### GEAR 3
    if (int(state.gear) == 3):
        if( engine_Rpm < 2000):
            gear = 2
        elif( engine_Rpm > 6500):
            gear = 4

    ### GEAR 4
    if ( int(state.gear) == 4):
        if( engine_Rpm < 2500):
            gear = 3
        elif( engine_Rpm > 6500):
            gear = 5

    ### GEAR 5
    if ( int(state.gear) == 5):
        if( engine_Rpm < 3000):
            gear = 4
        elif(engine_Rpm > 6500):
            gear = 6

    ### GEAR 6
    if ( int(state.gear) == 6):
        if( engine_Rpm < 3200):
            gear = 5
    
    return gear

