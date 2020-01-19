# Torcs Environment
from gym_torcs import TorcsEnv
import numpy as np

# Init Logging System
from Utilities.Init_system import start_sys,log_hyperparams
from Utilities.Params import HYPERPARAMS
# Utility Gear
from Utilities.Utils import gear_lookUp
# RL Algorithms
from Brain.Algorithms import DDPG 
from Brain.Controllers import Controls

# Remove Older Log Files and Create New Ones
paths = {
    'critic_path'           : "Logs/Weights_Critic",
    'policy_path'           : "Logs/Weights_Policy",
    'rp_path'               : "Logs/ReplayBuffers",
    'log_path'              : "Logs/Log",
    'actor_path_choosen'    : 'Best_Actor_Weights/actormodel.pth',
    'critic_path_choosen'   : 'Best Critic_Weights/criticmodel.pth',
    'rp_path_choosen'       : 'rp_0.pth',
    'policy_save_path'      : "Logs/Weights_Policy/policy.weight_",
    'critic_save_path'      : "Logs/Weights_Critic/critic.weight_",
    'RepBuff_save_path'     : "Logs/ReplayBuffers/rp_",
    'reward_save_path'      : 'Logs/Log/reward.csv'
}
start_sys(paths,if_scratch=True)

# Read all hyperparameters and network info and log them
log_hyperparams( HYPERPARAMS , 'Logs/Log/log','Logs/Log/reward')

# Create DDPG Agent
Agent = DDPG(HYPERPARAMS,paths)

# Torcs - Gym 
env = TorcsEnv(vision=False, throttle=True, gear_change=True)

# Training
for i in range(2000):

    if np.mod(i, 3) == 0:
        ob = env.reset(relaunch = True)
    else:
        ob = env.reset()

    s_t = np.hstack((ob.angle, ob.track, ob.trackPos, ob.speedX, ob.speedY, ob.speedZ, ob.wheelSpinVel/100.0, ob.rpm))

    total_rewards = 0

    for j in range(100000):
        a_t = Agent.decide(s_t,ob)

        # Add gear to actions
        action_gear = gear_lookUp(ob)

        action = [a_t[0][0],a_t[0][1],a_t[0][2],action_gear]
        
        # Apply action and observe next state 
        next_ob, r_t, done, info = env.step(action)
        s_t1 = np.hstack((next_ob.angle, next_ob.track, next_ob.trackPos, next_ob.speedX, next_ob.speedY, next_ob.speedZ, next_ob.wheelSpinVel/100.0, next_ob.rpm))

        Agent.train(s_t1,s_t,next_ob,a_t,r_t,done)

        s_t = s_t1
        ob = next_ob
        total_rewards = total_rewards + r_t

        if done:
            total_rewards = total_rewards.tolist()
            Agent.save_model(paths,i,total_rewards)
            break

    str1 = "Trial : [ {0} ] is completed with reward : [ {1} ] and lasted [ {2} ] steps.".format(i+1,total_rewards,j)
    log_name = paths['log_path'] + "/log" + ".txt"
    with open(log_name, 'a') as out:
        out.write(str1 + '\n')
    print(str1)

env.end()
print("================ Training is finished.Do not forget to check if your AI is planning to take over the world :) ========================")