import random
import numpy as np
import pickle

# RL object
class RL():
    def __init__(self, actions, epsilon, alpha, gamma):
        self.Q = {}

        self.A = actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.leng = leng

    # method to return the Q based on (state, action)
    def get_Q(self, state, action):
        # default 0
        if self.leng and state[-1] != 0:
            return self.Q.get((state, action), self.get_Q((state[:-1]+tuple([state[-1]-1])),action))
        else:
            return self.Q.get((state, action), 0.0)
    
    # set the Q
    def set_Q(self, Q):
        self.Q = Q

    # load the Q from the txt
    def load_Q(self, file_name):
        self.Q =  pickle.load(open(file_name, "rb"))
    
    # save Q to the txt 
    def save_Q(self, file_name):
        f = open(file_name, "wb")
        pickle.dump(self.Q, f)
        f.close()
    
    # get the action based on the state
    def get_A(self, state):
        if random.random() < self.epsilon:
            result = random.choice(self.A)
        else:
            q_list = [self.get_Q(state, a) for a in self.A]
            max_q = max(q_list)
            index = np.where(np.array(q_list) == max_q)
            result = self.A[random.choice(index[0])]
        return result

# extend the RL object to get the Sarsa object
class Sarsa(RL):   
    # method to update Q for (state, action)
    # this is the only difference between Qlearning and SARSA
    # Sarsa is updating by using the same action chosen by the getQ method
    def update_Q(self, state, action, new_state, new_action, reward):
        q = self.Q.get((state, action), None)
        if q is None:
            self.Q[(state, action)] = reward
        else:
            new_q = self.get_Q(new_state, new_action)
            self.Q[(state, action)] = q + self.alpha * (reward + self.gamma * new_q - q)
