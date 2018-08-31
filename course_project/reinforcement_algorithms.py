import humanize
from datetime import datetime
from functools import wraps
import numpy as np
from collections import defaultdict


def log_time(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        start_time = datetime.now()
        r = f(*args, **kwargs)
        end_time = datetime.now()
        print("Time Taken to run {time_take}".format(time_take=humanize.naturaltime(end_time-start_time).replace(' ago','')))
        return r
    return wrapped

class RL(object):
    def __init__(self, agent, number_of_episodes, discount_factor, learning_rate, max_tries, epsilon=None):
        self.agent = agent
        self.number_of_episodes = number_of_episodes
        self.discount_factor = discount_factor
        self.max_tries = max_tries
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.q = defaultdict(lambda: np.zeros(self.agent.num_actions))

    def sarsa_learning(self):
        rewards = []
        for i in range(self.number_of_episodes):
            self.agent.reset(self.agent.rat_init)
            (row, col, mode) = self.agent.state
            state = (row, col)
            action = self.get_epsilon_greedy_action(state)
            timestemps = 0
            while True:
                env_state, reward, status = self.agent.act(action)
                new_state = (self.agent.state[0], self.agent.state[1])
                new_action = self.get_epsilon_greedy_action(new_state)
                self.q[state][action] += self.learning_rate * (reward + (self.discount_factor * self.q[new_state][new_action]) - self.q[state][action])
                if status == 'win' or timestemps >= self.max_tries:
                    rewards.append(self.agent.total_reward)
                    break
                timestemps += 1
                state = new_state
                action = new_action
                # if timestemps%40 == 0:
                #     # self.agent.show()
                #     print("Episode : {episode} , timesteps: {timestemps}".format(episode=i,timestemps=timestemps))
            if i%10 == 0:
                print("Max reward for {episode} episodes = {max_reward}".format(episode=i, max_reward=max(rewards)))
        return rewards

    def q_learning(self):
        rewards = []
        for i in range(self.number_of_episodes):
            self.agent.reset(self.agent.rat_init)
            rewards += self.run_episode()
            if i%10 == 0:
                print("Max reward for {episode} episodes = {max_reward}".format(episode=i, max_reward=max(rewards)))
        return rewards

    def run_episode(self):
        rewards = []
        (row, col, mode) = self.agent.state
        state = (row, col)
        timestemps = 0
        while True:
            action = self.get_argmax_action(self.q, state=state)
            env_state, reward, status = self.agent.act(action)
            new_state = (self.agent.state[0], self.agent.state[1])
            self.q[state][action] += self.learning_rate * (
                        reward + (self.discount_factor * max(self.q[new_state])) - self.q[state][action])
            timestemps += 1
            state = new_state
            self.agent.show()
            if status == 'win' or timestemps >= self.max_tries:
                rewards.append(self.agent.total_reward)
                self.agent.show()
                break
            if timestemps % 40 == 0:
                self.agent.show()
                pass
        return rewards

    def play_q_learning(self):
        self.agent.reset(self.agent.rat_init)
        no_of_steps = 0
        while True:
            (row, col, mode) = self.agent.state
            action = self.get_argmax_action(self.q, (row, col))
            env_state, reward, status = self.agent.act(action)
            no_of_steps += 1
            self.agent.show()
            if no_of_steps%20 ==0:
                print("Number of step {no_of_steps} reward {reward}".format(no_of_steps=no_of_steps, reward=self.agent.total_reward))
                self.agent.show()
            if status == 'win':
                self.agent.show()
                print("Number of steps mouse has taken to reach cheese : {no_of_steps}".format(no_of_steps=no_of_steps))
                break

    def expected_sarsa(self):
        rewards = []
        for i in range(self.number_of_episodes):
            self.agent.reset(self.agent.rat_init)
            (row, col, mode) = self.agent.state
            state = (row, col)
            action = self.get_epsilon_greedy_action(state)
            timestemps = 0
            while True:
                env_state, reward, status = self.agent.act(action)
                new_state = (self.agent.state[0], self.agent.state[1])
                new_action = self.get_epsilon_greedy_action(new_state)
                self.q[state][action] += self.learning_rate * (
                            reward + (self.discount_factor * self.q[new_state][new_action]) - self.q[state][action])
                if status == 'win' or timestemps >= self.max_tries:
                    rewards.append(self.agent.total_reward)
                    break
                timestemps += 1
                state = new_state
                action = new_action
                if timestemps % 100 == 0:
                    self.agent.show()
            if i % 10 == 0:
                print("Max reward for {episode} episodes = {max_reward}".format(episode=i, max_reward=max(rewards)))
        return rewards

    def get_epsilon_greedy_action(self, state, epsilon=None):
        if not self.epsilon:
            print("No epsilon value provided")
            return
        if epsilon:
            self.epsilon = epsilon
        m = np.argmax(self.q[state])
        probabilites = []
        max_prob = 1-self.epsilon+(self.epsilon/self.agent.num_actions)
        prob = self.epsilon/self.agent.num_actions
        for i in range(self.agent.num_actions):
            if i == m:
                probabilites.append(max_prob)
            else:
                probabilites.append(prob)
        return np.random.choice(range(self.agent.num_actions), p=probabilites)

    def get_argmax_action(self, q, state):
        m = np.amax(q[state])
        indices = np.nonzero(q[state] == m)[0]
        return np.random.choice(indices)

