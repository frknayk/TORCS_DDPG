# TORCS_DDPG
Learning how to drive from scratch with DDPG algorithm 

### How To Use This REPO

***train.py*** 
- Main code that training happens. Inside this code, you need to enter paths of
log files, paths of past trained weights of actor and critic and the path of replay memory.
- This is useful, because you can train your AI, from a specific weights and a point of memory.

***inference.py***
- Inference code for the agent with given trained paths.

***params.py***
- Parameters of the DDPG algorithm, neural networks and noise parameters are given here. So you do not need to
  change them inside of functions every time when you start new training, just change them from here. 
  
***/Brain***
- Inside this folder neural network arhitecture and algorithms are given. Furthermore low level PID or infinite horizon 
  lqr controllers are also implemented here. LQR controller uses dynamic bicycle model which is implemented in the 
  'Dynamics.py' script.
