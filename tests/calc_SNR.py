import numpy as np
from scipy.constants import *
import cmath
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from matplotlib import colors

'''
miscellaneous functions
'''


# conversion from dB to linear
def db2lin(val_db):
    val_lin = 10 ** (val_db / 10)
    return val_lin


# cosine over degree
def cosd(val):
    return math.cos(val * pi / 180)


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


class MIMO:
    def __init__(self, init_ptx, init_RBS, init_TBS, tr_antennas, rx_antennas):

        self.freq = 28e9  # 28 GHz
        self.d = 0.5  # relative element space
        # transmitter and receiver location
        self.X_range = 108
        self.X_angle = 0

        self.lmda = c / self.freq  # c - speed of light, scipy constant
        self.P_tx = init_ptx  # dBm
        self.TBS = init_TBS  # 0 radians, range [-pi, pi]
        self.RBS = init_RBS  # 0 raidans, range [-pi, pi]
        self.N_tx = tr_antennas  # no. of transmitting antennas
        self.N_rx = rx_antennas  # no. of receiving antennas

        x = self.X_range * math.cos(self.X_angle*pi/180)
        y = self.X_range * math.sin(self.X_angle*pi/180)
        X = [x, y]  # row list of x,y

        self.X_t = X[0]
        self.X_r = X[1]
        self.Dist = np.linalg.norm(np.array(self.X_t) - np.array(self.X_r))
        self.tau_k = self.Dist / c

    def Transmit_Energy(self, ptx):
        self.df = 75e3  # carrier spacing frequency
        self.nFFT = 2048  # no. of subspace carriers

        self.T_sym = 1 / self.df
        self.B = self.nFFT * self.df
        self.P_tx = ptx
        Es = db2lin(self.P_tx) * (1e-3/self.B)
        # self.P_tx += beta
        return Es

    def Channel(self):

        FSL = 20 * np.log10(self.Dist) + 20 * np.log10(self.freq) - 147.55  # db, free space path loss
        channel_loss = db2lin(-FSL)
        g_c = np.sqrt(channel_loss)
        h = g_c * cmath.exp(-1j * (pi / 4))  # LOS channel coefficient
        return h

    def array_factor(self, ang, N):
        x = np.arange(0, N)
        # y = np.array([1 / np.sqrt(N) * np.exp(1j * 2 * pi * (self.d / self.lmda) * math.sin(ang) * k) for k in x])
        y = np.zeros((N, 1), dtype=np.complex)
        for k in x:
            y[k] = 1 / np.sqrt(N) * np.exp(1j * 2 * pi * (self.d / self.lmda) * cmath.sin(ang) * k)
            y[k] = complex(np.around(y[k].real, decimals=4), np.around(y[k].imag, decimals=4))

        # print("y: {0}".format(y))
        return y

    def Antenna_Array(self, RBS, TBS, level):
        alpha = 0  # relative rotation between transmit and receiver arrays

        if self.X_r > 0:
            self.theta_tx = math.acos(self.X_t / self.Dist)
        else:
            self.theta_tx = -1 * math.acos(self.X_t / self.Dist)

        self.phi_rx = self.theta_tx - pi + alpha

        a_tx = self.array_factor(self.theta_tx, self.N_tx)
        a_rx = self.array_factor(self.phi_rx, self.N_rx)

        self.RBS = RBS
        self.TBS = TBS
        # print("RBS: {0}, TBS: {1}".format(self.RBS, self.TBS))
        w_mat = self.Communication_Vector(self.RBS, self.N_tx, level)  # transmit unit norm vector

        f_mat = self.Communication_Vector(self.TBS, self.N_rx, level)  # receive unit norm vector

        return a_tx, a_rx, self.N_tx, self.N_rx, w_mat, f_mat

    def Noise(self):
        N0dBm = -174 #mW/Hz
        N0 = db2lin(N0dBm) * (10 ** -3) #in WHz-1
        return N0

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

        # function to define tranmsit or receive unit norm vector

    def Communication_Vector(self, ang, n, level):
        if level == 0:
            phi_mv = [0]
        else:
            l = [-1, 0, 1]
            omega = [sind(ang) + x / math.pow(3, level) for x in l]
            phi_mv = [asind(x) for x in omega]
            # print("Omega: {0}, {1}, {2}".format(omega, ang, phi_mv))

        n_xyz = [1, n, 1]

        D_el = np.eye(n_xyz[2])
        # print(n,int(math.pow(3,level)))
        # Na = np.min([n, int(math.pow(3,level))])
        D_az = np.zeros((n, len(phi_mv)), dtype=np.complex)
        # print(D_az.shape)
        for k in range(len(phi_mv)):
            # print(D_az[k].shape)
            D_az[:, k] = self.root_beam(phi_mv[k], level, n).ravel()
            # print("Vector values: {0}".format(D_az[:][k]))
        return np.kron(D_az, D_el)


    def Calc_SNR(self, ptx, rbs, tbs, level):

        Es = self.Transmit_Energy(ptx)  # beta_pair[0])
        h = self.Channel()
        #print("channel coefficient: {0}".format(h))
        antenna_ret = self.Antenna_Array(rbs, tbs, level)
        a_tx, a_rx, N_tx, N_rx, w_mat, f_mat = antenna_ret
        #print("w_mat: {0}".format(w_mat))
        #print("f_mat: {0}".format(f_mat))
        N0 = self.Noise()

        rssi_val = np.zeros((f_mat.shape[1], w_mat.shape[1]))  # (tx levels, rx_levels)
        for i in range(rssi_val.shape[1]):
            for j in range(rssi_val.shape[0]):

                wRF = np.zeros((self.N_rx, 1), dtype=np.complex)
                wRF[:, 0] = w_mat[:, i].ravel()

                fRF = np.zeros((self.N_rx, 1), dtype=np.complex)
                fRF[:, 0] = f_mat[:, j].ravel()

                # print("wRF: {0}".format(np.matmul(np.matmul(wRF.conj().T,H_k), fRF)[0,0]))
                r_f = h*np.matmul(np.matmul(np.matmul(wRF.conj().T, a_rx),a_tx.conj().T), fRF)#need not be done-*np.sqrt(N_tx*N_rx)
                #print("channel coefficient: {0}, wRF: {1}, fRF: {2}".format(h, wRF, fRF))
                    #np.sqrt(ptx / self.nFFT) * c_k * np.matmul(np.matmul(wRF.conj().T, H_k), fRF)[0, 0] * a[
                    #n - 1] * np.exp(-1j * 2 * pi * n * self.df * self.tau_k) + N0_f
                rssi_val[j, i] += ((r_f.real) ** 2 + (r_f.imag) ** 2)
                #print("RSSI_val({1},{2}): {0}".format(rssi_val[i,j], i, j))
                #break

        best_RSSI_val = np.max(rssi_val)
        SNR = Es * best_RSSI_val / N0
        #print("best RSSI_val: {0}".format(best_RSSI_val))
        #print("Measure SNR: {0}, Es: {1}, No: {2}, indmax: {3}".format(SNR, Es, N0, np.argmax(rssi_val)))
        return SNR

def Action_Space_Mapping(actions):

    parameter_count = len(actions.keys())
    parameter_list = []
    for key in actions.keys():
        par_range = actions[key]#[actions.keys[i]]
        parameter_list.append(list(range(par_range[0],par_range[1]+1,par_range[2])))


    #creates a list of all possible tuples from given lists of action values
    action_val_tuples = [list(x) for x in np.array(np.meshgrid(*parameter_list)).T.reshape(-1,len(parameter_list))]
    action_key_list = list(np.arange(len(action_val_tuples)))

    action_values = dict(zip(action_key_list,action_val_tuples))
    #print("action_values: {0}".format(action_values))
    return action_values

if __name__ == '__main__':
    Actions = {
        #'ptx': [20, 44, 1],
        'RBS': [-60, 60, 5],  # [-24 * pi / 216, 24 * pi / 216, 6 * pi / 216]
        'TBS': [-60, 60, 5],
        'BeamWidth': [1,3,1]
    }

    action_values = Action_Space_Mapping(Actions)

    best_ptx = 46#Actions['ptx'][0]
    best_rbs = Actions['RBS'][0]
    best_tbs = Actions['TBS'][0]
    best_level = Actions['BeamWidth'][0]
    #ptx_values = np.arange(Actions['ptx'][0], Actions['ptx'][1]+1, Actions['ptx'][2])
    rbs_values = np.arange(Actions['RBS'][0], Actions['RBS'][1]+1, Actions['RBS'][2])
    tbs_values = np.arange(Actions['TBS'][0], Actions['TBS'][1]+1, Actions['TBS'][2])


    snr_list=[]
    maxSNR = -120
    SNR_counts ={}
    for i in range(-120,61):
        SNR_counts[i] = 0
    #mimo = MIMO(Actions['ptx'][0], Actions['RBS'][0], Actions['TBS'][0], 16,16)
    init_ptx = best_ptx
    init_rbs = 0
    init_tbs = 180
    Ntx = 16
    Nrx = 16
    mimo = MIMO(init_ptx,init_rbs, init_rbs, Ntx, Nrx)
    #level=1
    goal_count=0
    for k in action_values.keys():
        #ptx, rbs, tbs = action_values[k]
        rbs, tbs, level = action_values[k]

        #snr = mimo.Calc_SNR(ptx, rbs, tbs)
        snr = mimo.Calc_SNR(best_ptx, rbs, tbs, level)

        logSNR = np.around(10 * np.log10(snr), decimals=5)

        SNR_counts[int(np.around(logSNR,2))] +=1


        snr_list.append(logSNR)
        if logSNR > maxSNR:
            maxSNR = logSNR
            #best_ptx, best_rbs, best_tbs = ptx, rbs, tbs
            best_rbs, best_tbs, best_level = rbs, tbs, level
        #break
    print("Max SNR reached: {0}, best_ptx: {1}, best_rbs: {2}, best_tbs: {3}, best_level: {4}".format(maxSNR, best_ptx, best_rbs, best_tbs, best_level))
    c=np.array(snr_list)
    print("Goal Count: {0}".format(goal_count))
    #c_int = np.array([int(np.around(x)) for x in snr_list])


    '''
    Display the parameters where SNR is maximum
    '''
    count=0
    for i in range(len(c)):
        if c[i] == maxSNR:
            count+=1
            print("Optimal parameter {0}: index: {1}, values: {2}".format(count, i, action_values[i]))


    # c = np.random.standard_normal(25)
    fig = plt.figure()
    '''
    x_surf, y_surf = np.meshgrid(rbs_values, tbs_values)
    z_surf = np.outer(np.ones(len(ptx_values)).T, ptx_values)
    print(x_surf.shape, y_surf.shape, z_surf.shape)

    k_surf = np.outer(np.ones(len(ptx_values)*len(rbs_values)*len(tbs_values)).T,c)
    color_dimension = k_surf
    minn, maxx = color_dimension.min(), color_dimension.max()
    norm = colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap='jet')
    m.set_array([])
    fcolors = m.to_rgba(color_dimension)

   
    ax = fig.add_subplot(2,1, 1, projection='3d')

    surf = ax.plot_surface(x_surf, y_surf, z_surf,rstride=1,cstride=1,cmap=cm.coolwarm, vmin=minn, vmax=maxx, shade=False)#, vmin=minn,vmax=maxx

    #sp = ax.scatter(ptx_values, rbs_values, tbs_values, s=20, c=c)
    #plt.colorbar(surf)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    '''
    ax = fig.add_subplot(3,1,1)
    ax.grid()
    ax.plot(action_values.keys(), c)
    ax.set_title('RSSI Exhaustive Search\n Ptx: {0},RBS: {1}, TBS: {2}, Level: {3}, Ntx: {4}, NRx: {5}'.format(best_ptx, Actions['RBS'], Actions['TBS'],Actions['BeamWidth'], Ntx, Nrx))
    ax.set_ylabel('SNR values')
    ax.set_xlabel('Action Pairs')
    #ax.scatter(x, y, z, c=c, cmap=plt.hot())


    ax2 = fig.add_subplot(3,1,3)
    ax2.grid()
    ax2.plot(SNR_counts.keys(),SNR_counts.values())
    ax2.set_title("Histogram of SNR counts")
    ax2.set_ylabel("Counts")
    ax2.set_xlabel("SNR values")

    plt.show()




    '''
    Scatter Plot
    '''
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sp = ax.scatter(ptx_values, rbs_values, tbs_values, s=20, c=c)
    plt.colorbar(sp)
    plt.show()
    '''

