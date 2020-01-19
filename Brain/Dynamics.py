import numpy as np

"""
Car dynamic bicycle model ( can be used as model for LQR controller )
"""
class Car_Bicycle(object):
    def __init__(self,v_ref,Ts):
        ### TORCS car1-trb-1 model
        self.C_f = 80000    # Cornerng Stiffness of Front Tire
        self.C_r = 80000    # Cornerng Stiffness of Rear Tire
        self.L_f = 1.27     # Longitudunal Distance from COG of Front Tire
        self.L_r = 1.37     # Longitudunal Distance from COG of Rear Tire
        self.M = 1150       # Mass
        self.I_z = 2000     # Moment of Inertia 
        self.vx = v_ref     # Initial Car Heading Dir. vel.
        
        ### System Dynamics
        self.A21 = -(2*self.C_f + 2*self.C_r)/(self.M * self.vx)
        self.A22 = (2*self.C_f + 2*self.C_r)/(self.M)
        self.A23 = (-2*self.C_f*self.L_f + 2*self.C_r*self.L_r)/(self.M * self.vx)

        self.A41 = (-2*self.C_f*self.L_f + 2*self.C_r*self.L_r)/(self.I_z*self.vx)
        self.A42 = (2*self.C_f*self.L_f - 2*self.C_r*self.L_r)/(self.I_z)
        self.A43 = (-2*self.C_f*(self.L_f**2) + 2*self.C_r*(self.L_r**2))/(self.I_z*self.vx)


        self.A = np.array([ 
            [0,1,0,0],
            [0,self.A21,self.A22,self.A23],
            [0,0,0,1],
            [0,self.A41,self.A42,self.A43] ])

        self.B = np.array([ [0],
        [(2*self.C_f)/self.M],
        [0],
        [(2*self.C_f*self.L_f)/self.M] ])


        self.e_yaw_prev = 0.0
        self.e_latDev_prev = 0.0
        self.dt = Ts

    def remake_dynamics(self):
        self.A21 = -(2*self.C_f + 2*self.C_r)/(self.M * self.vx)
        self.A23 = (-2*self.C_f*self.L_f + 2*self.C_r*self.L_r)/(self.M * self.vx)

        self.A41 = (-2*self.C_f*self.L_f + 2*self.C_r*self.L_r)/(self.I_z*self.vx)
        self.A43 = (-2*self.C_f*(self.L_f**2) + 2*self.C_r*(self.L_r**2))/(self.I_z*self.vx)

        self.A = np.array([ 
            [0,1,0,0],
            [0,self.A21,self.A22,self.A23],
            [0,0,0,1],
            [0,self.A41,self.A42,self.A43] ])


    def make_state(self,yaw_error,latDev_error):
        
        e1 = latDev_error 
        e1_der = (latDev_error - self.e_latDev_prev)/self.dt
        e2 = yaw_error
        e2_der = (e2 - self.e_latDev_prev)/self.dt

        state = np.array( [e1,e1_der,e2,e2_der])

        self.e_yaw_prev = e2
        self.e_latDev_prev = e1

        return state