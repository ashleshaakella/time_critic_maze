import numpy as np
from collections import defaultdict


class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.alpha = 0.1
        self.gamma = 1

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # select an action based on self.Q

        m = np.amax(self.Q[state])
        indices = np.nonzero(self.Q[state] == m)[0]
        return np.random.choice(indices)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        next_action = self.select_action(next_state)
        # sarsa
        # self.Q[state][action] += (self.alpha * (reward + (self.gamma*self.Q[next_state][next_action])) -
        #                           self.Q[state][action])
        # sarsamax (Q-learning)
        # self.Q[state][action] += self.alpha * (reward + (self.gamma * max(self.Q[next_state])) - self.Q[state][action])
