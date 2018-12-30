import numpy as np


class Bandit:

    def __init__(self, sd):
        self.q_star = np.random.normal()
        self.sd = sd

    def get_sample(self):
        return np.random.normal(self.q_star, self.sd)

    def __str__(self):
        return "Bandit with q*star: " + self.q_star + "and sd: " + self.sd


class KArmedBandit:

    def __init__(self, k, sd=1):
        self.k = k
        self.bandit = {}
        self.sd = sd
        for i in range(1, k+1):
            self.bandit[i] = Bandit(sd)

    def reward(self, action):
        return self.bandit[action].get_sample()

    def __str__(self):
        return_val = ""
        for k, v in self.bandit.items():
            return_val += k + "---" + v + "\n"
        return return_val
