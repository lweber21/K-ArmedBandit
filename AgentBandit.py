
import Bandit
import numpy as np
import matplotlib.pyplot as plt
import random


class AgentBandit:

    def __init__(self, k=4):
        self.k = k
        self.Qval = {}
        self.N = {}
        self.reset()
        self.bd = Bandit.KArmedBandit(k)
        self.rewardTally = []
        self.optimalChoice = []
        self.x = []

    def reset(self):
        self.Qval = {}
        self.N = {}
        for i in range(1, self.k + 1):
            self.Qval[i] = 0
            self.N[i] = 0

    def train(self, eps, trial):
        self.reset()

        for i in range(trial):
            self.x.append(i)
            if np.random.uniform() < eps:
                action = random.randint(1, self.k)
            else:
                action = max(self.Qval.keys(), key=(lambda j: self.Qval[j]))

            reward, optimal = self.bd.reward(action)

            self.rewardTally.append(reward)
            self.optimalChoice.append(optimal)

            self.N[action] += 1

            self.Qval[action] = self.Qval[action] + (1.0 / self.N[action]) * (reward - self.Qval[action])

        return self.rewardTally

    def get_rewards(self):
        return self.rewardTally

    def plot_results(self):

        plt.plot(self.rewardTally)
        plt.show()


agent = AgentBandit()
#display trained values
print(agent.bd)
agent.train(0.1, 10000)
print(agent.Qval)
#print(agent.optimalChoice)
#agent.plot_results()



#
