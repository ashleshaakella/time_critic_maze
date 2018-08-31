import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt


import torch
torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

env = gym.make('CartPole-v0')
print('observation space:', env.observation_space)
print('action space:', env.action_space)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(self, s_size=4, h_size=16, a_size=2):
        """Neural network that encodes the policy.

        Params
        ======
            s_size (int): dimension of each state (also size of input layer)
            h_size (int): size of hidden layer
            a_size (int): number of potential actions (also size of output layer)
        """
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


class SolveCartPole(object):

    @staticmethod
    def random_solution():
        env = gym.make('CartPole-v0')

        state = env.reset()
        img = plt.imshow(env.render(mode='rgb_array'))
        for t in range(1000):
            action = env.action_space.sample()
            img.set_data(env.render(mode='rgb_array'))
            plt.axis('off')
            state, reward, done, _ = env.step(action)
            if done:
                print('Score: ', t + 1)
                break
        env.close()

    @staticmethod
    def reinforce(n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
        """PyTorch implementation of the REINFORCE algorithm.

        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            gamma (float): discount rate
            print_every (int): how often to print average score (over last 100 episodes)
        """
        env.seed(0)

        policy = Policy().to(device)
        optimizer = optim.Adam(policy.parameters(), lr=1e-2)

        scores_deque = deque(maxlen=100)
        scores = []
        for i_episode in range(1, n_episodes + 1):
            saved_log_probs = []
            rewards = []
            state = env.reset()
            for t in range(max_t):
                action, log_prob = policy.act(state)
                saved_log_probs.append(log_prob)
                state, reward, done, _ = env.step(action)
                rewards.append(reward)
                if done:
                    break
            scores_deque.append(sum(rewards))
            scores.append(sum(rewards))

            discounts = [gamma ** i for i in range(len(rewards) + 1)]
            R = sum([a * b for a, b in zip(discounts, rewards)])

            policy_loss = []
            for log_prob in saved_log_probs:
                policy_loss.append(-log_prob * R)
            policy_loss = torch.cat(policy_loss).sum()

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            if i_episode % print_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            if np.mean(scores_deque) >= 195.0:
                print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                           np.mean(scores_deque)))
                break

        return scores


if __name__ == "__main__":
    scores = SolveCartPole().reinforce()





