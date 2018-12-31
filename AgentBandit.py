
import Bandit
import numpy as np
import matplotlib.pyplot as plt
import random


class AgentBandit:

    def __init__(self, k=4, eps=0, trial=1000):
        self.k = k
        self.eps = eps
        self.trial = trial
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

    def train(self):
        self.reset()

        for i in range(self.trial):
            self.x.append(i)
            if np.random.uniform() < self.eps:
                action = random.randint(1, self.k)
            else:
                action = max(self.Qval.keys(), key=(lambda j: self.Qval[j]))

            reward, optimal = self.bd.reward(action)

            self.rewardTally.append(reward)
            self.optimalChoice.append(optimal)

            self.N[action] += 1

            self.Qval[action] = self.Qval[action] + (1.0 / self.N[action]) * (reward - self.Qval[action])

        return self.rewardTally, self.optimalChoice

    def get_rewards(self):
        return self.rewardTally

    def get_optimal(self):
        return self.optimalChoice

    def plot_results(self):

        plt.plot(self.rewardTally)
        plt.show()


#Run Simulation on 2000 different bandits

nBandits = 2000
episodes = 1000
arms = 10
results = np.zeros((nBandits, episodes))
optimalVal = np.zeros((nBandits, episodes))
for i in range(nBandits):

    agent = AgentBandit(arms, 0.1, episodes)
    results[i, :], optimalVal[i, :] = agent.train()


avg_reward = np.mean(results, axis=0)
x_axis = np.arange(1, episodes + 1)
opt = 100*np.sum(optimalVal, axis=0)/nBandits


plt.plot(x_axis, avg_reward)
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.show()

plt.plot(x_axis, opt)
plt.xlabel("Episode")
plt.ylabel("% Optimal Choice")
plt.show()






