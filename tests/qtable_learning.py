import gym

import gym_rf

# from gym_rf.envs.rf_env import MIMO
from scipy.constants import *
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm

class Model:
    def __init__(self, env, alpha):
        self.env = env
        self.Q = np.zeros([self.env.observation_space.n, self.env.action_space.n])
        for i in range(self.env.observation_space.n):
            for j in range(self.env.action_space.n):
                self.Q[i,j] = -1*alpha

    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[s,:])


def play_one_episode(model, eps, gamma, alpha):
    observation = model.env.reset()
    done = False
    iters = 0
    totalreward = 0
    # action = env.action_space.sample()
    while (not done and iters < 300):
        action = model.sample_action(observation, eps)
        # print("action:{0}".format(action))
        prev_observation = observation

        observation, reward, done, info = model.env.step(action)
        #print("reward: {0}".format(reward))
        model.Q[prev_observation, action] += alpha*(reward + gamma*np.max(model.Q[observation, :])- model.Q[prev_observation, action])

        totalreward += reward

        iters += 1
        if ((iters + 1) % 100) == 0:
            print("Play one episode: {0} iters are done".format(iters + 1))
        #break
    return totalreward

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
    #print("action_values: {0}".format(action_values))
    return action_values

if __name__ == '__main__':
    env = gym.make('rf-v0')
    alpha = 0.1
    model = Model(env, alpha)
    gamma = 0.99
    N = 200

    rwd_list=[]
    rwds=[]
    for n in range(N):
        eps = 1.0 / np.sqrt(n + 1)
        #state = env.reset()
        rwd = play_one_episode(model, eps, gamma, alpha)

        rwd_list.append(rwd)
        if (len(rwd_list) % 100) == 0:
            rwds.append(np.mean(rwd_list[-100: ]))
    #print(rwd_list)
    print("Score over time: {0}".format(sum(rwd_list) /N))

    print("Final Q-table: {0}".format(model.Q))

    fig = plt.figure(figsize=plt.figaspect(2.))
    fig.suptitle('Q table learning plots')

    ax = fig.add_subplot(4, 1, 1)
    print('Average Reward: {0}'.format(np.array(rwds).mean()))
    ax.plot(rwds)
    ax.grid()
    ax.set_ylabel('Rewards')
    # plt.show()
    # ax = plt.subplot(3,1,3)
    #Actions = {
        # 'ptx': [24, 30, 2],
    #    'RBS': [-60, 60, 5],  # [-24 * pi / 216, 24 * pi / 216, 6 * pi / 216]
    #    'TBS': [-60, 60, 5]
    #}
    #states = {
    #    'SNR': [-120, 32, 1]  # -120dB to 65dB
    #}
    Actions = model.env.Actions
    states = model.env.Observations
    # rbs_values = np.arange(Actions['RBS'][0], Actions['RBS'][1] + 1, Actions['RBS'][2])
    # tbs_values = np.arange(Actions['TBS'][0], Actions['TBS'][1] + 1, Actions['TBS'][2])
    action_values = Custom_Space_Mapping(Actions)
    state_values = Custom_Space_Mapping(states)
    # print(len(action_values.values()), len(model.Q[1,:]))
    # print(model.Q.shape[0])
    action_values_list = action_values.keys()
    ax2 = fig.add_subplot(4, 1, 2)
    for n in range(model.Q.shape[0]):
        if np.max(model.Q[n, :]) > 0:
            ax2.plot(action_values_list, model.Q[n, :], label="n=%d" % (n,))
    ax2.grid()
    ax2.set_xlabel('actions')
    ax2.set_ylabel('Q values')

    '''
       Print the learnt optimal path in reaching the desired SNR
    '''
    rev_state_values = dict((v[0], k) for k, v in state_values.items())
    observation = rev_state_values[-2]
    done = False
    model.env.reset()
    count = 0
    # print(observation)
    # print(state_values)
    # print(state_values[observation])
    SNR_path = [state_values[observation][0]]

    max_snr_state = rev_state_values[60]  # model.env.observation_space.n -1 #maximum state
    while ((observation < max_snr_state) and (count < 10)):
        predict = model.Q[observation]
        # print('Final predict values: {0} '.format(predict.argmax()))
        action = np.argmax(predict)
        # action = predict.argmax()
        previous_observation = observation
        observation, reward, done, info = model.env.step(action)
        print(
            "Previous Obs: {0}, action: {1}, action_values: {2}, Observation: {3}, done: {4}".format(state_values[previous_observation][0],
                                                                                 action, action_values[action],
                                                                                 state_values[observation][0], done))
        SNR_path.append(state_values[observation][0])
        # max_snr_state = np.max((max_snr_state, observation))
        count += 1
    # print(np.max(SNR_path))
    max_snr_state = 0
    # for s in range(model.Q.shape[0]):
    #    if (np.max(model.Q[s,:]) != 0) and (s > max_snr_state):
    #        max_snr_state = s
    # print(max_snr_state)
    goal_state = rev_state_values[60]
    # print(state_values)
    # print(rev_state_values)
    print(model.Q.shape)
    print(np.max(model.Q[goal_state, :]))#, np.max(model.Q[182]))
    print("SNR 30 best action: {0}, action_value: {1}, SNR 30 Q-value: {2}".format(np.argmax(model.Q[goal_state, :]),
                                                                                   action_values[np.argmax(
                                                                                       model.Q[goal_state, :])],
                                                                                   np.max(model.Q[goal_state, :])))
    # print("Max state learnt: {0}, action: {1}, action_value: {2}, Q-value: {4}".format(state_values[max_snr_state], np.argmax(model.Q[max_snr_state, :]), action_values[np.argmax(model.Q[max_snr_state, :])], np.max(model.Q[max_snr_state,:])))

    ax3 = fig.add_subplot(4, 1, 4)
    ax3.plot(np.arange(len(SNR_path)), SNR_path)
    ax3.grid()
    ax3.set_xlabel('step')
    ax3.set_ylabel('SNR path')

    plt.show()

