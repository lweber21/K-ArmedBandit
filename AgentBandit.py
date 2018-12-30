
import Bandit
import numpy as np
import matplotlib.pyplot as plt
import random


class AgentBandit:

    def __init__(self, k=4):
        self.k = k
        self.eps = 0
        self.trial = 1000
        self.Qval = {}
        self.N = {}
        for i in range(1, k + 1):
            self.Qval[i] = 0
            self.N[i] = 0
        self.bd = Bandit.KArmedBandit(k)
        self.rewardTally = []

    def simulate(self, eps, trial):
        self.eps = eps
        self.trial = trial
        print(self.trial)

        for i in range(self.trial):
            if np.random.uniform() < self.eps:
                action = random.randint(1, self.k + 1)
            else:
                action = max(self.Qval.keys(), key=(lambda j: self.Qval[j]))

            reward = self.bd.reward(action)

            self.rewardTally.append(reward)

            self.N[action] += 1

            self.Qval[action] += self.Qval[action] + (1.0 / self.N[action]) * (reward - self.Qval[action])

            return self.rewardTally

    def get_rewards(self):
        return self.rewardTally

    def plot_results(self):
        x = range(0, self.trial)
        plt.plot(x, self.rewardTally)


agent = AgentBandit(4)
print(agent)

#agent.plot_results()


