import scipy.linalg
import numpy as np
import math

# Import Dynamic - Bicycle Car Model
from Brain.Dynamics import Car_Bicycle

# Import utilitiy functions
from Utilities.Utils import  gear_lookUp

"""
Low level controllers : PID and Infinite Horizon LQR
"""
class Controls(object):
    def __init__(self,HYPERPARAMS):
        
        ### Longitudunal PID Controller 
        self.vel_pid = HYPERPARAMS['vel_pid']

        # It works great while reference speed is '100' km/h
        self.reference_speed = 125.0

        self.Hz = HYPERPARAMS['Hz']
        self.Ts = float(1/self.Hz)

        # LQR Weight Matrices
        self.Q = HYPERPARAMS['LQR_Q']
        self.R = HYPERPARAMS['LQR_R']

        # Low Level Controller Saturation Limits
        self.STEER_MAX    = HYPERPARAMS['STEER_MAX']
        self.THROTTLE_MAX = HYPERPARAMS['THROTTLE_MAX']
        self.THROTTLE_MIN = HYPERPARAMS['THROTTLE_MIN']

        # Set speed pid controller
        self.PID_speed = PID(self.vel_pid,self.reference_speed,self.Ts)
        self.PID_speed.sat_max = self.THROTTLE_MAX # Saturation upper limit
        self.PID_speed.sat_min = self.THROTTLE_MIN # Saturation lower limit

        ### Init Car Dynamical Bicycle Model
        self.car = Car_Bicycle(self.reference_speed,self.Ts)

        # LQR for lateral controller
        self.lqr = LQR()

        # System dynamics (dynamical bicycle model)
        self.A = self.car.A
        self.B = self.car.B

        self.lqr_gain_cont,_,_ = self.lqr.lqr(self.A,self.B,self.Q,self.R)
        self.lqr_gain_cont = np.array(self.lqr_gain_cont)
        
    def actuators(self,ob,references):

        # Reference Signals Produced by RL
        yaw_ref,letDev_ref,speed_ref = references

        if (ob.speedX > 1):
            self.car.vx = ob.speedX
            self.car.remake_dynamics()
            self.A = self.car.A
            self.B = self.car.B
            self.lqr_gain_cont,X,eigVals = self.lqr.lqr(self.A,self.B,self.Q,self.R)
            self.lqr_gain_cont = np.array(self.lqr_gain_cont)

        ### Calculate Steering 
        # Policy tries to predict track's curvature actually
        yaw_error    = yaw_ref - ob.car_yaw
        # Policy tries also tries to predict lateral deviation
        latDev_error = letDev_ref - ob.trackPos 

        state = self.car.make_state(yaw_error,latDev_error)

        action_steer = np.matmul(self.lqr_gain_cont,state)
        action_steer = math.radians(action_steer)
        action_steer = saturation(action_steer,self.STEER_MAX,-self.STEER_MAX)      


        ### Calculate Throttle
        self.PID_speed.reference = speed_ref
        action_throttle = self.PID_speed.out(ob.speedX)

        ### Brake Calculated By Brake-Assist in TORCS side
        action_brake = ob.getBrake

        ### Gear Calculated By a Look-Up Table
        action_gear = gear_lookUp(ob)

        if(action_brake == 0.0 and self.PID_speed.error<0):
            action_brake = math.fabs(action_throttle)
            if( action_brake>0) : 
                action_brake = 1
                action_throttle = 0

        return [action_steer,action_throttle,action_brake,action_gear]

class PID(object):
    def __init__(self,K_PID,ref,dt):
        self.Kp = K_PID[0]
        self.Ki = K_PID[1]
        self.Kd = K_PID[2]
        self.error = 0
        self.error_int = 0
        self.error_prev = 0
        self.reference = ref
        self.dt = dt
        self.sat_max = np.inf
        self.sat_min = -np.inf
        self.normalize_term = self.reference*1.0
        self.anti_windup_sat = 0.25
        self.debug_flag = False

    def out(self,output):
        self.error = self.reference - output
        self.error = self.error/self.normalize_term
        self.error_int += self.error

        if(self.error_int > self.anti_windup_sat):
            self.error_int = self.anti_windup_sat
        der_error = (self.error - self.error_prev)/self.dt
        
        self.error_prev = self.error

        PID_Kp = self.Kp*self.error
        PID_Ki = self.Ki*self.error_int
        PID_Kd = self.Kd*der_error

        if self.debug_flag :
            print("Velocity Error   : ",self.error)
            print("PID-Kp term      : ",PID_Kp)
            print("PID-Ki term      : ",PID_Ki)

        U =  PID_Kp + PID_Ki + PID_Kd

        return saturation(U,self.sat_max,self.sat_min)

class LQR(object):
    def lqr(self,A,B,Q,R):
        """Solve the continuous time lqr controller.

        dx/dt = A x + B u

        cost = integral x.T*Q*x + u.T*R*u
        """
        #ref Bertsekas, p.151

        #first, try to solve the ricatti equation
        X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))

        #compute the LQR gain
        K = np.matrix(scipy.linalg.inv(R)*(B.T*X))

        eigVals, eigVecs = scipy.linalg.eig(A-B*K)

        return K, X, eigVals
    
    def dlqr(self,A,B,Q,R):
        """Solve the discrete time lqr controller.

        x[k+1] = A x[k] + B u[k]

        cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
        """
        #ref Bertsekas, p.151

        #first, try to solve the ricatti equation
        X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))

        #compute the LQR gain
        K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))

        eigVals, eigVecs = scipy.linalg.eig(A-B*K)

        return K, X, eigVals

def saturation(U,sat_max,sat_min):
    if(U  > sat_max):
        U = sat_max
    elif( U  < sat_min ):
        U = sat_min
    return U