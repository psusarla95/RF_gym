'''
author: psusarla
email: praneeth.susarla@oulu.fi
date: 25-12-2018
'''

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from scipy.constants import *
import cmath
import math

'''
miscellaneous functions
'''


# conversion from dB to linear
def db2lin(val_db):
    val_lin = 10 ** (val_db / 10)
    return val_lin


# cosine over degree
def cosd(val):
    return cmath.cos(val * pi / 180)


# sine over degree
def sind(val):
    return cmath.sin(val * pi / 180)

# asin in degree
def asind(val):
    #return 180/pi*math.asin(val)
    #return np.degrees(np.sinh(val))
    c1 = cmath.asin(val)
    c2 = complex(math.degrees(c1.real), math.degrees(c1.imag))
    return c2

#deg2rad for complex number
def deg2rad(val):
    l = [val.real*cmath.pi/180, val.imag*cmath.pi/180]
    c1 = complex(np.around(l[0], decimals=4), np.around(l[1], decimals=4))
    return c1

# acosine in degree
def acosd(val):
    return np.degrees(np.sinh(val))

'''
MIMO class

- Defines the MIMO model of the system
- Uses a LOS communication model and a free space path loss
- Considers Ptx, RBS, TBS, NRx, NTx as inputs to the system
- Computes Transmit Energy, Antenna Response vectors, Channel coefficients, SNR estimations etc.
'''
class MIMO:

    '''
    __init__(init_ptx, init_RBS, init_TBS, tr_antenna, rx_antenna)
    ptx - power transmitter level in dB
    RBS - Receiver Beam steering angle in degree
    TBS - Transmitter Beam steering angle in degree
    tx_antennas - No. of antenna elements at TX unit
    rx_antenns - No. of antenna elements at RX unit

    - Consider a fixed location with X_range=108, X_angle = 0
    - Consider a mmwave frequency for the system, freq=28GHz
    - Consider a fixed relative antenna element space, d=0.5

    '''
    def __init__(self, init_ptx, init_RBS, init_TBS, tr_antennas, rx_antennas, xrange, xangle):

        self.freq = 28e9  # 28 GHz
        self.d = 0.5  # relative element space
        # transmitter and receiver location
        self.X_range = xrange
        self.X_angle = xangle

        self.lmda = c / self.freq  # c - speed of light, scipy constant
        self.P_tx = init_ptx  # dBm
        self.TBS = init_TBS  # 0 radians, range [-pi, pi]
        self.RBS = init_RBS  # 0 radians, range [-pi, pi]
        self.N_tx = tr_antennas  # no. of transmitting antennas
        self.N_rx = rx_antennas  # no. of receiving antennas

        x = self.X_range * math.cos(self.X_angle*pi/180)
        y = self.X_range * math.sin(self.X_angle*pi/180)
        X = [x, y]  # row list of x,y

        self.X_t = X[0]
        self.X_r = X[1]
        self.Dist = np.sqrt(self.X_t**2 + self.X_r**2)#np.linalg.norm(np.array(self.X_t) - np.array(self.X_r))
        self.tau_k = self.Dist / c

    '''
    Es = Transmit_Energy(ptx)
    ptx - power transmitter level in dB
    Es  - Transmit Energy in W
    
    - Computes the transmit energy of the MIMO model based on the fixed Ptx
    
    Assumptions:
        - Carrier Spacing frequency considered is 75KHz
        - No. of subspace carriers - 2048
     
    '''
    def Transmit_Energy(self, ptx):
        self.df = 60*1e3#75e3  # carrier spacing frequency
        self.nFFT = 1200#2048  # no. of subspace carriers

        self.T_sym = 1 / self.df
        self.B = self.nFFT * self.df
        self.P_tx = ptx
        Es = db2lin(self.P_tx) * (1e-3/self.B)
        return Es

    '''
    h = Channel()
    h - channel coefficient
    - Computes the channel coefficient
    - Channel is assumed to have a 'free space path loss' between the TX and RX
    '''
    def Channel(self):

        FSL = 20 * np.log10(self.Dist) + 20 * np.log10(self.freq) - 147.55  # db, free space path loss
        channel_loss = db2lin(-FSL)
        g_c = np.sqrt(channel_loss)
        h = g_c * cmath.exp(-1j * (pi / 4))  # LOS channel coefficient
        return h

    '''
    y = array_factor(ang, N)
    Computes response vectors of the antenna unit 
    
    Parameters:
    ang - Angle in degree
    N - No. of antenna elements in the unit
    
    Output:
    y - Array Response vector
    
    '''
    def array_factor(self, ang, N):
        x = np.arange(0, N)
        y = np.zeros((N, 1), dtype=np.complex)
        for k in x:
            y[k] = 1 / np.sqrt(N) * np.exp(1j * 2 * pi * (self.d / self.lmda) * cmath.sin(ang) * k)
            y[k] = complex(np.around(y[k].real, decimals=4), np.around(y[k].imag, decimals=4))
        return y

    '''
    a_tx, a_rx, N_tx, N_rx, w_mat, f_mat  = array_factor(RBS, TBS, rlevel, tlevel)
    Computes response vectors, unit normalization vectors at both transmitter and receiver antenna units 

    Parameters:
    RBS - Receiver Beam Steering angle in degree
    TBS - Transmitter Beam Steering angle in degree
    rlevel - Receiver Beam Width level [0,1,2,3]
    tlevel - Transmitter Beam Width level [0,1,2,3]
    

    Output:
    a_tx - Transmitter Array Response vector
    a_rx - Receiver Array Response vector
    N_tx - No. of transmitter antenna elements
    N_rx - No. of receiver antenna elements
    w_mat - Receiver unit normalization vector
    f_mat - Transmit unit normalization vector
    
    Assumptions:
    The relative rotation between Tx and RX arrays is fixed, alpha= 0

    '''


    def Antenna_Array(self, RBS, TBS, rlevel, tlevel):
        alpha = 0  # relative rotation between transmit and receiver arrays

        if self.X_r > 0:
            self.theta_tx = math.acos(self.X_t / self.Dist)
        else:
            self.theta_tx = -1 * math.acos(self.X_t / self.Dist)

        self.phi_rx = self.theta_tx - pi + alpha

        #transmit array response vector
        a_tx = self.array_factor(self.theta_tx, self.N_tx)

        #receiver array response vector
        a_rx = self.array_factor(self.phi_rx, self.N_rx)

        self.RBS = RBS
        self.TBS = TBS
        # print("RBS: {0}, TBS: {1}".format(self.RBS, self.TBS))
        w_mat = self.Communication_Vector(self.RBS, self.N_tx, rlevel)  # receive unit norm vector

        f_mat = self.Communication_Vector(self.TBS, self.N_rx, tlevel)  # transmit unit norm vector

        return a_tx, a_rx, self.N_tx, self.N_rx, w_mat, f_mat

    '''
    N0 = Noise()
    Models the noise present in the channel
    
    Output:
    N0 - Noise model
    
    '''
    def Noise(self):
        N0dBm = -174 #mW/Hz
        N0 = db2lin(N0dBm) * (10 ** -3) #in WHz-1
        return N0

    '''
    y = root_beam(phi_m, level, N)
    
    - Computes the root beam of the antenna unit
    - Useful in unit normal vector estimations
    
    Parameters:
    phi_m - root beam angle in degree
    level - beam width level [0,1,2,3]
    N - no. of antenna elements in the unit
    
    Output:
    y - root beam vector 
    '''
    def root_beam(self, phi_m, level, N):
        phi_m = deg2rad(phi_m)  # converting deg to radians
        Na = np.min([N, int(math.pow(3, level))])

        x = np.arange(0, Na)
        y = np.zeros((N, 1), dtype=np.complex)
        # print(y.shape)
        for k in x:
            # print("phi_m: {0}".format(phi_m))
            y[k] = np.exp(1j * 2 * pi * 0.5 * cmath.sin(phi_m) * k)
        # print(y.shape)
        return y

    '''
    y = Communication_vector(ang, n, level)
    
    - function to define tranmsit or receive unit norm vector
    
    Parameters:
    ang - beam steering angle in degree
    n - no. of antenna elements
    level - beam width level [0,1,2,3]
    
    Output:
    y - matrix of Unit normal vectors along columns
    '''

    def Communication_Vector(self, ang, n, level):
        if level == 0:
            phi_mv = [0]
        else:
            l = [-1, 0, 1]
            omega = [sind(ang) + x / math.pow(3, level) for x in l]
            phi_mv = [asind(x) for x in omega]


        n_xyz = [1, n, 1]
        D_el = np.eye(n_xyz[2])
        D_az = np.zeros((n, len(phi_mv)), dtype=np.complex)
        # print(D_az.shape)
        for k in range(len(phi_mv)):

            D_az[:, k] = self.root_beam(phi_mv[k], level, n).ravel()
            # print("Vector values: {0}".format(D_az[:][k]))
        return np.kron(D_az, D_el)

    '''
    SNR = Calc_SNR(ptx, rbs, tbs, rlevel, tlevel)

    - Estimates SNR based on the given TX and RX parameters

    Parameters:
    ptx - Transmitter power in dB
    rbs - Receiver Beam steering angle in degree
    tbs - Transmitter Beam steering angle in degree
    rlevel - receiver beam width level [0,1,2,3]
    tlevel - transmitter beam width level [0,1,2,3]

    Output:
    SNR - SNR computation from RSSI Energy in W
    '''
    def Calc_SNR(self, ptx, rbs, tbs, rlevel, tlevel):

        Es = self.Transmit_Energy(ptx)  # beta_pair[0])
        h = self.Channel()

        antenna_ret = self.Antenna_Array(rbs, tbs, rlevel, tlevel)
        a_tx, a_rx, N_tx, N_rx, w_mat, f_mat = antenna_ret
        N0 = self.Noise()

        rssi_val = np.zeros((f_mat.shape[1], w_mat.shape[1]))  # (tx levels, rx_levels)
        for i in range(rssi_val.shape[1]):
            for j in range(rssi_val.shape[0]):

                wRF = np.zeros((self.N_rx, 1), dtype=np.complex)
                wRF[:, 0] = w_mat[:, i].ravel()

                fRF = np.zeros((self.N_rx, 1), dtype=np.complex)
                fRF[:, 0] = f_mat[:, j].ravel()

                #LOS Communication model for computing RSSI Energy
                r_f = h*np.matmul(np.matmul(np.matmul(wRF.conj().T, a_rx),a_tx.conj().T), fRF)#*np.sqrt(N_tx*N_rx)
                rssi_val[j, i] += ((r_f.real) ** 2 + (r_f.imag) ** 2)

                #print("RSSI_val({1},{2}): {0}".format(rssi_val[i,j], i, j))

        best_RSSI_val = np.max(rssi_val)
        self.SNR = Es * best_RSSI_val / N0

        return self.SNR

    def Calc_Rate(self, stepcount, Tf):
        Tf = Tf * 1e-3  # for msec
        ktf = np.ceil(Tf / self.T_sym)
        Tf_time = ktf * self.T_sym

        #print("SNR while calculating rate: {0}".format(self.SNR))
        #levels = 3
        rate = self.B*(1-(stepcount)*self.T_sym/Tf_time)*np.log2(1+self.SNR)*1e-9 #in Gbit/s
        return rate

    def Calc_RateOpt(self, stepcount, Tf, ptx):
        Es = self.Transmit_Energy(ptx)
        h = self.Channel()
        N0 = self.Noise()

        # transmit array response vector
        a_tx = self.array_factor(self.theta_tx, self.N_tx)

        # receiver array response vector
        a_rx = self.array_factor(self.phi_rx, self.N_rx)

        SNROpt = h*np.sqrt(self.N_rx)*np.matmul(np.matmul(np.matmul(a_rx.conj().T, a_rx), a_tx.conj().T),  a_tx)*np.sqrt(self.N_tx)
        SNROpt = Es*((SNROpt.real)**2 + (SNROpt.imag)**2)/N0
        SNROpt = SNROpt[0][0]

        Tf = Tf*1e-3 #for msec
        ktf = np.ceil(Tf/self.T_sym)
        Tf_time = ktf*self.T_sym
        RateOpt = self.B*(1-stepcount*self.T_sym/Tf_time)*np.log2(1+SNROpt)*1e-9 #in Gbit/s

        #print("SNROpt: {0}, RateOpt: {1}".format(SNROpt, RateOpt))


        return RateOpt

'''

###################
Custom Space Mapping
####################

Example:
actions = {
    ['RBS'] :[0,2,1],
    ['TBS'] :[0,2,1]
    }
Custom_Space_Mapping(actions) =
    { 0:[0,0],1:[0,1],2:[0,2],3:[1,0],4:[1,1], 5:[1,2], 6:[2,0], 7:[2,1], 8: [2,2]}

'''
def Custom_Space_Mapping(actions):

    parameter_count = len(actions.keys())
    parameter_list = []
    for key in actions.keys():
        par_range = actions[key]#[actions.keys[i]]
        parameter_list.append(list(range(par_range[0],par_range[1]+1,par_range[2])))


    #creates a list of all possible tuples from given lists of action values
    action_val_tuples = [list(x) for x in np.array(np.meshgrid(*parameter_list)).T.reshape(-1,len(parameter_list))]
    action_key_list = list(np.arange(len(action_val_tuples)))

    action_values = dict(zip(action_key_list,action_val_tuples))

    return action_values


''''
####################################
    MAIN CLASS ENVIRONMENT
####################################
RFBeamEnv - RF Beam Environment

Model Characteristics:
- Considers a MIMO model with mmwave frequency
- Considers a fixed Ptx and chooses Beam steering vectors- Receiver Beam Steering (RBS),
  Transmitter Beam Steering Vectors (TBS), Beam Width Level (level) as the main parameters for this RF Beam model 

RL Method: Q-learning
- states - integer SNR values (dB); actions- (RBS,TBS,RBeamWidth,TBeamWidth);
- Observation Threshold state: 30 dB (SNR), Observation goal state: 60 dB (SNR)
                5*exp((curr_state-goal_state)/10) if curr_state>= observation_thr_state;
  Rewards = {   5                                 if curr_state = observation_goal_state;
                -1                                 otherwise      
- RBS span -[-60,60,5] (deg), TBS span-[-60,60,5](deg), RBeamWidth span- [1,3,1] (level), TBeamWidth span- [1,3,1] (level), SNR span-[-120,60,1] (dB)
- Observation space - [0,1,2,.....,179] -> [-120, -119, -118,......,60]
- Action space - [0,1,2,.......5624] -> [(-60,-60,1,1), ......(RBS,TBS,RBeamWidth,TBeamWidth).......(60,60,3,3)]

- Transmit Power= 46 dB, N_tx= 16, N_rx=16
'''

class RFBeamEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.Actions = {
            # 'ptx': [24, 30, 2],
            'RBS': [-50, 50, 5],  # [-24 * pi / 216, 24 * pi / 216, 6 * pi / 216]
            'TBS': [-50, 50, 5],
            'RBeamWidth': [1,3,1],
            'TBeamWidth': [1,3,1]
        }
        self.Observations = {
            'SNR': [-30, 50, 1] #-120dB to 65dB
        }

        self.observation_values = Custom_Space_Mapping(self.Observations)
        self.rev_observation_values = dict((v[0],k) for k,v in self.observation_values.items())
        self.num_observations = len(self.observation_values.keys())
        self.min_state = self.observation_values[0][0]#minimum SNR state
        #self.max_state = self.observation_values[self.num_observations-1][0]#maximum SNR state
        self.state_threshold = 28 #good enough SNR state
        self.N_tx = 16 #Num of transmitter antenna elements
        self.N_rx = 16 #Num of receiver antenna elements
        self.count = 0
        self.ptx =  30
        self.level = 1
        self.state = None

        # Initializing parameters with their minimal values
        #self.xrange=700
        #self.xangle=40

        #self.mimo = MIMO(self.ptx,self.Actions['RBS'][0],self.Actions['TBS'][0], self.N_tx, self.N_rx)

        self.action_values = Custom_Space_Mapping(self.Actions)
        self.num_actions = len(self.action_values.keys())
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(self.num_observations)
        self.seed()
        self.viewer=None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_distance(self, xrange, xangle, goal_state):
        # Initializing parameters with their minimal values
        self.xrange= xrange
        self.xangle= xangle
        self.max_state=goal_state
        self.mimo = MIMO(self.ptx, 0, 180, self.N_tx, self.N_rx, self.xrange, self.xangle)

    def set_state(self, s):
        self.state = s

    def get_current_state(self):
        return self.state

    '''
    state, reward, done, {} - step(action)
    - A basic function prototype of an Env class under gym
    - This function is called every time, the env needs to be updated into a new state based on the applied action
    
    Parameters:
    action - the action tuple applied by the RL agent on Env current state
    
    Output:
    state - The new/update state of the environment
    reward - Env reward to RL agent, for the given action
    done - bool to check if the environment goal state is reached
    {} - empty set
    
    '''
    def step(self, action):
        #check the legal move first and then return its reward
        #if action in self.actions[self.current_state]:
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        prev_state = self.state

        SNR = self.mimo.Calc_SNR(self.ptx,self.action_values[action][0], self.action_values[action][1], self.action_values[action][2], self.action_values[action][3])

        '''
        The below logic is to ensure that we select the optimal values of parameters like power transmitter and TBS, only within the chosen SNR range
        '''
        logSNR = int(np.around(10*np.log10(SNR),decimals=2)) #considering int values of SNR

        if logSNR > self.max_state:
            SNR_state = self.max_state
        elif logSNR < self.min_state:
            SNR_state = self.min_state
        else:
            SNR_state = logSNR

        self.count+=1
        done = self.game_over(SNR_state) #This is the crucial step for algo to reach the final state early

        '''
        if done:
            reward = 15
            print("Reached Final state: {0}, Actions Taken: {1}, steps: {2}".format(SNR_state, self.action_values[action], self.count))
        elif SNR_state <= prev_state:
            reward = -35
        elif SNR_state >= self.state_threshold:
            reward = 15*np.exp((SNR_state - self.state_threshold) / 10)  # shaping reward function
        else:
            reward=0
        '''
        if done:
            reward = 10
            print("Reached Final state: {0}, Actions Taken: {1}, steps: {2}".format(SNR_state, self.action_values[action], self.count))
        elif SNR_state <= prev_state:
            reward = -40
        elif SNR_state < self.state_threshold:
            reward = -5
        else:
            reward = 10* np.exp((SNR_state - self.max_state) / 10)  # shaping reward function

        #Mapping back from SNR range to observation space
        state = self.rev_observation_values[SNR_state]
        #print("calculated logSNR: {0}, logSNR_state: {1}".format(SNR_state, state))
        self.state = state
        return self.state, reward, done, {}

    '''
    game_state = game_over(s)
    - Function to check if the agent has reached its goal in the environment
    
    Parameters:
    s - current state of the environment
    
    Output:
    game_state {    False       if s < goal_state
                    True        if s = goal_state
    '''
    def game_over(self, s):
        if s >= self.max_state:
            return True
        else:
            return False

    '''
    reset()
    - Resets the environment to its default values
    - Prototype of the gym environment class  
    '''

    def reset(self):
        # Note: should be a uniform random value between starting 4-5 SNR states
        observation= self.observation_space.sample()
        self.state = self.observation_values[observation][0]

        #self.mimo = MIMO(self.ptx,self.Actions['RBS'][0], self.Actions['TBS'][0], 4, 4)
        #self.mimo = MIMO(self.ptx, 0, 180, self.N_tx, self.N_rx, self.xrange, self.xangle)
        self.count=0
        return self.state

    '''
    test_reset(Xrange, Xangle, action_val)
    - Reset the environment to a new MIMO model during test phase of RL model
    
    Parameters:
    xrange - distance between transmitter and receiver radio units
    xangle - angle between transmitter and reciever radio units
    action_val - Randomly generated action tuple (RBS,TBS, RBeamWidth, TBeamWidth) to be applied on RFBeamEnv with new MIMO model 
    
    '''

    def test_reset(self, xrange, xangle, action_val, goal_state):

        #New mimo model
        #self.mimo = MIMO(self.ptx, 0, 180, self.N_tx, self.N_rx, xrange, xangle)
        self.set_distance(xrange, xangle, goal_state)
        #self.mimo.X_range = xrange
        #self.mimo.X_angle = xangle
        self.count=0
        #random action
        #action = self.action_space.sample()
        SNR = self.mimo.Calc_SNR(self.ptx, action_val[0], action_val[1], action_val[2], action_val[3])

        logSNR = int(np.around(10 * np.log10(SNR), decimals=2))  # considering int values of SNR

        if logSNR > self.max_state:
            SNR_state = self.max_state
        elif logSNR < self.min_state:
            SNR_state = self.min_state
        else:
            SNR_state = logSNR

        #print("SNR state calculated: {0}".format(SNR_state))
        state = self.rev_observation_values[SNR_state]

        return state, action_val

    def get_Rate(self, stepcount, Tf):
        rateOpt = self.mimo.Calc_RateOpt(stepcount,Tf,self.ptx)
        rate = self.mimo.Calc_Rate(stepcount, Tf)
        return rate, rateOpt

    def render(self, mode='human', close=False):
        pass

    def action_value(self, action):
        return self.action_values[action]
