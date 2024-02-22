from tqdm import tqdm
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

def greedy_action(Q,s,nb_actions):
    Qsa = []
    for a in range(nb_actions):
        sa = np.append(s,a).reshape(1, -1)
        Qsa.append(Q.predict(sa))
    return np.argmax(Qsa)


def rf_fqi(S, A, R, S2, D, iterations, nb_actions, gamma, disable_tqdm=False):
    nb_samples = S.shape[0]
    Qfunctions = []
    SA = np.append(S,A,axis=1)
    for iter in tqdm(range(iterations), disable=disable_tqdm):
        if iter==0:
            value=R.copy()
        else:
            Q2 = np.zeros((nb_samples,nb_actions))
            for a2 in range(nb_actions):
                A2 = a2*np.ones((S.shape[0],1))
                S2A2 = np.append(S2,A2,axis=1)
                Q2[:,a2] = Qfunctions[-1].predict(S2A2)
            max_Q2 = np.max(Q2,axis=1)
            value = R + gamma*(1-D)*max_Q2
        Q = RandomForestRegressor()
        Q.fit(SA,value)
        Qfunctions.append(Q)
    return Qfunctions


def collect_samples(env, horizon, disable_tqdm=False, print_done_states=False):
    s, _ = env.reset()
    S = []
    A = []
    R = []
    S2 = []
    D = []
    for _ in tqdm(range(horizon), disable=disable_tqdm):
        a = env.action_space.sample()
        s2, r, done, trunc, _ = env.step(a)
        S.append(s)
        A.append(a)
        R.append(r)
        S2.append(s2)
        D.append(done)
        if done or trunc:
            s, _ = env.reset()
            if done and print_done_states:
                print("done!")
        else:
            s = s2
    S = np.array(S)
    A = np.array(A).reshape((-1,1))
    R = np.array(R)
    S2= np.array(S2)
    D = np.array(D)
    return S, A, R, S2, D

class ProjectAgent:
    def __init__(self, Qvalue=None) -> None:
        self.Qvalue = Qvalue
        self.n_actions = 4

    def act(self, observation, use_random=False):
        action = greedy_action(self.Qvalue, observation, nb_actions=self.n_actions)
        return action

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.Qvalue, f)

    def load(self):
        with open("src/model.pkl", "rb") as f:
            self.Qvalue = pickle.load(f)

if __name__ == "__main__":
    gamma = .9
    nb_iter = 10
    nb_actions = env.action_space.n
    S,A,R,S2,D = collect_samples(env, int(1e4))
    Qfunctions = rf_fqi(S, A, R, S2, D, nb_iter, nb_actions, gamma)
    Qvalue = Qfunctions[-1]
    agent = ProjectAgent(Qvalue)
    agent.save("model.pkl")