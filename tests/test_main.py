import gym

import gym_rf

#from gym_rf.envs.rf_env import MIMO
from scipy.constants import *
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

'''
env = gym.make('rf-v0')
env.seed(0)
env.reset()
s = env.action_space.sample()
print(s)
env.step(s)
'''

#env = gym.make('rf-v0')
#print(env.observation_space)
#print(env.action_space)
'''
def SpaceDict_toKey(space_dict):
    v1 = space_dict['ptx'][0]
    v2 = space_dict['rbs'][0]
    return tuple([v1, v2])
'''

class Feature_Transformer:
    def __init__(self, env, n_components=500):
        observation_samples = np.array([env.observation_space.sample() for x in range(10000)])
        scaler = StandardScaler()
        scaler.fit(observation_samples.astype('float'))

        #convert a state to feature representation
        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=3.0, n_components=n_components)),
            #("rbf2", RBFSampler(gamma=.5, n_components=n_components)),
        ])

        featurizer.fit(scaler.transform(observation_samples.astype('float')))

        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        scaled = self.scaler.transform(observations)
        s = self.featurizer.transform(scaled)
        #print('came here: {0}'.format(type(scaled)))
        return s

class Model:
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        #self.action_values = Action_Space_Mapping(actions)
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate=learning_rate)
            # This partial fit allows us to use optimistic initial value methods of exploration
            model.partial_fit(feature_transformer.transform([self.env.reset()]), [0])
            self.models.append(model)

    def predict(self, obs):
        #print('came here: {0}'.format(type(obs)))
        X = self.feature_transformer.transform([obs])

        result = np.stack([m.predict(X) for m in self.models]).T
        assert (len(result.shape) == 2)
        return result

    def update(self, s, a, G):

        X = self.feature_transformer.transform([s])
        #print('X: {0}'.format(np.array(G)))
        assert(len(X.shape) == 2)
        self.models[a].partial_fit(X, np.array([G]).ravel())

    def sample_action(self, s , eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))

    def learnt_actions(self, action):
        action_values = self.env.action_values(action)
        return action_values
'''
class Model:
    def __init__(self, env):
        self.env = env
        self.Q = {}


    def Check_Observation(self, obs):

        print('observation: {0}'.format(obs))
        #print(self.Q.keys())
        #print(self.env.observation_space.contains(obs))
        if not bool(self.Q):
            self.Q[obs[0]] = {}
            return True
        elif self.Q.get(obs[0]) is not None:
            return True
        elif (self.Q.get(obs[0]) is None) and (self.env.observation_space.contains(obs)):
            self.Q[obs[0]] = {}
            action = str(self.env.action_space.sample()) #type(action) is a space dict (opeanai gym)
            mod_action = SpaceDict_toKey(action) #the result type(action) is a tuple (python)
            #print(self.Q)
            #print(obs, action, type(action))
            self.Q[obs[0]][mod_action] = 0
            return True
        else:
            return False

    def sample_action(self, observation, eps=0.1):
        # epsilon-soft method (also known as epsilon-greedy from now on)
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            if bool(self.Q[observation[0]]): #Q[obs] dict is not empty
                return self.max_dict(observation[0])[0]
            elif self.env.observation_space.contains(observation):
                #self.Q[observation[0]]={}
                action = self.env.action_space.sample()
                mod_action = SpaceDict_toKey(action) #type(action2) is a tuple (python)
                self.Q[observation[0]][mod_action] = 0
                return action
            else:
                return 'observation error'

    def max_dict(self, obs):
        max_key = None
        max_val = float('-inf')

        for k, v in self.Q[obs].items():
            if v > max_val:
                max_key = k
                max_val = v
    
        if max_key is None:
            action = self.env.action_space.sample()
            max_key = action
            max_val = 0
            self.Q[obs][action]= max_val


        return max_key, max_val

    def predict(self, observation):
        if bool(self.Q[observation[0]]):
            return self.max_dict(observation)[0]

    def update(self, s, a, G, alpha):
        self.Q[s][a] += alpha*G
'''




def play_one(model, eps, gamma, alpha):
    observation = model.env.reset()
    done=False
    iters = 0
    totalreward = 0
    #action = env.action_space.sample()

    while (not done and iters<150):
        action = model.sample_action(observation, eps)
        #print("action:{0}".format(action))
        prev_observation = observation

        observation, reward, done, info = model.env.step(action)

        #if done:
        #    print("its done")
        #    break
        #if done:
        #    reward = -40
        #print('After: {0}'.format(model.Q))
        #next_action = model.predict(observation)
        #print("No. of models: {0}".format(len(model.models)))
        #print('Next action: {0}'.format(next_action[0][0]))
        #assert(model.env.action_space.contains(next_action) == True)#len(next_action) == len(model.env.action_space))
        #G = reward + gamma*model.Q[observation][next_action]
        predict=model.predict(observation)[0]
       # print("Length of predict: {0}".format(predict))
        G = reward + gamma *np.max(predict)
        model.update(prev_observation, action, G)

        #if reward == 20:
        totalreward += reward
        iters += 1
        if((iters +1) % 100) == 0:
            print("Play one episode: {0} iters are done".format(iters+1))
        break
    return totalreward



if __name__ == '__main__':
    env = gym.make('rf-v0')
    ft = Feature_Transformer(env)
    model = Model(env, ft, "constant")
    gamma = 0.99
    alpha=0.1
    N = 15
    totalrewards = np.empty(N)
    costs = np.empty(N)
    print(model.env.observation_space.n, model.env.action_space.n)
    for n in range(N):
        eps = 1.0 / np.sqrt(n + 1)
        totalreward = play_one(model, eps, gamma, alpha)

        totalrewards[n] = totalreward

        #print("Reward for episode {0}: {1}".format(n,totalreward))
        #if n % 100 == 0:
        #    print("episode: ", n, "total reward: ", totalreward, "eps: ", eps, "avg reward (last 100):",
        #          totalrewards[max(0, n - 100):(n + 1)].mean())
        break
    print("avg reward for last 100 episodes: ", totalrewards.mean())
    #print("total steps: ", totalrewards.sum())

    plt.subplot(3,1,1)
    #print('plot: {0}'.format(totalrewards.shape))
    plt.plot(totalrewards)
    plt.title("Rewards")

    '''
       Print the learnt optimal path in reaching the desired SNR
    '''

    observation = np.array([14])
    done = False
    model.env.reset()
    count=0
    SNR_path = [observation]
    
    while (not done and count<1):
        predict = model.predict(observation)[0]
        #print('Final predict values: {0} '.format(predict.argmax()))
        action = np.argmax(predict)
        #action = predict.argmax()
        previous_observation = observation
        observation, reward, done, info = model.env.step(action)
        print("Previous Obs: {0}, action: {1}, Observation: {2}".format(previous_observation,action, observation))
        SNR_path.append(observation)
        count+=1

    plt.subplot(3, 1, 3)
    plt.plot(np.arange(len(SNR_path)) + 1, SNR_path, '.-')
    plt.axis([1, 20, 15, 45])
    plt.ylabel('SNR states')
    plt.xlabel('Steps')
    plt.grid()

    plt.show()