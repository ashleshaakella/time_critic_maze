import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

import torch.optim as optim

import torch.autograd as autograd
import numpy as np
from collections import defaultdict

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
np.random.seed(325)


class MemoryReplayBuffer(object):

    def __init__(self):
        self.data = []

    def add(self, x):
        self.data.append(x)

    def refresh(self):
        self.data = []


class HierarchicalSemiDeepQLearning(object):
    """
    HierarchicalSemiDeepQLearning has one meta-controller which is DQN
    and each controller is a Q-learning agent

    """

    def __init__(self, maze, number_of_episodes, discount_factor, learning_rate, max_tries, epsilon, batch_size=100):
        meta_controller_agent = maze.meta_controller
        controller_agents = maze.agents

        # Construct meta-controller and controller
        self.maze = maze
        self.number_of_episodes = number_of_episodes
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.max_tries = max_tries
        self.epsilon = epsilon
        self.batch_size = batch_size
        num_of_meta_controller_actions = int(math.pow(len(controller_agents), 2)) - 1
        self.num_of_controller_actions = maze.agents[0].num_actions
        self.meta_controller = DQN(in_features=self.maze.maze_matrix.size, out_features=num_of_meta_controller_actions,
                                   learning_rate=learning_rate).type(dtype)
        self.target_meta_controller = DQN(in_features=self.maze.maze_matrix.size,
                                          out_features=num_of_meta_controller_actions,
                                          learning_rate=learning_rate).type(dtype)

        self.memory = MemoryReplayBuffer()

        self.meta_controller_agent = meta_controller_agent
        self.controller = [Qlearning(agent_maze=agent) for agent in controller_agents]

    def play(self):
        self.run_episode(episode_num=0, update_qs=False)

    def q_learning(self):
        rewards = []
        rewards_per_agent = defaultdict(lambda: [])
        for i in range(self.number_of_episodes):
            # reset agents
            [agent.reset(agent.rat_init) for agent in self.maze.agents]
            _reward = self.run_episode(episode_num=i)
            for agent_num, reward in _reward.items():
                print("Max reward for {agent} agent is {max_reward}".format(agent=agent_num,max_reward=reward))

    def run_episode(self, episode_num, update_qs=True):
        rewards = defaultdict(lambda: 0)
        agent_reached_goal = [False for i in self.controller]
        meta_controller_timesteps = 0
        agent_timesteps = np.zeros(shape=len(self.maze.agents))
        states = list(map(lambda x: (x[0], x[1]), map(lambda x: x.state, self.maze.agents)))

        # analysis variables
        num_of_time_q_state_is_updated_per_controller = [defaultdict(lambda: np.zeros(self.num_of_controller_actions))
                                                         for i in range(len(self.controller))]

        while True:
            if all([agent_reached_goal[i] for i in range(len(self.controller))]):
                break

            # collect the action from meta controller
            meta_current_state = deepcopy(self.meta_controller_agent.current_state)
            if np.random.random() >= self.epsilon:
                action = self.meta_controller_agent.random_action_index
            else:
                action = np.argmax(self.meta_controller.get_q_values(meta_current_state))
            meta_controller_action = self.meta_controller_agent.get_action(
                action) if meta_controller_timesteps >= self.batch_size else np.ones(len(self.controller))

            actions = []
            for i, agent in enumerate(self.controller):
                if not agent_reached_goal[i]:
                    # play only if the agent didnt reach goal
                    if meta_controller_action[i] == 1:
                        actions.append(self.get_argmax_action(agent.q, state=states[i]))
                    else:
                        actions.append(None)
                else:
                    actions.append(None)

            agents_new_state = []
            for i, agent in enumerate(self.controller):
                if actions[i] is not None:
                    env_state, reward, status = agent.maze.act(actions[i])
                    rewards[i] += reward
                    new_state = (agent.maze.state[0], agent.maze.state[1])
                    if update_qs is True:
                        agent.q[states[i]][actions[i]] += self.learning_rate * (
                                reward + (self.discount_factor * max(agent.q[new_state])) -
                                agent.q[states[i]][actions[i]])
                        num_of_time_q_state_is_updated_per_controller[i][states[i]][actions[i]] += 1
                    states[i] = new_state
                    agent_timesteps[i] += 1
                    if status == 'win' or agent_timesteps[i] >= self.max_tries:
                        # rewards[i].append(agent.total_reward)
                        self.maze.show()
                        agent_reached_goal[i] = True
                agents_new_state.append((agent.maze.state[0], agent.maze.state[1]))
            time.sleep(0.1)
            self.maze.show()

            self.meta_controller_agent.update_state([i.maze.rat for i in self.controller])
            meta_controller_timesteps += 1
            meta_controller_reward = self.meta_controller_agent.reward(agents_positions=agents_new_state)
            rewards[len(self.controller)] += meta_controller_reward
            # get state, action, newstate and reward for the meta_controller
            self.memory.add((meta_current_state, meta_controller_action, self.meta_controller_agent.current_state,meta_controller_reward))

            collision = self.meta_controller_agent.collision(agents_positions=agents_new_state)
            # if collision:
            #     if update_qs:
            #         self.train_hdqn(batch_inputs=self.memory.data)
            #         self.memory.refresh()
            #     break

            if meta_controller_timesteps % self.batch_size == 0 and update_qs:
                self.train_hdqn(batch_inputs=self.memory.data)
                self.memory.refresh()

        import pickle
        with open(str(episode_num)+'.pik', 'wb') as io:
            k = []
            for i in num_of_time_q_state_is_updated_per_controller:
                k.append(dict([(k, v.tolist()) for (k, v) in i.items()]))
            pickle.dump(k, io)
        return rewards

    def get_argmax_action(self, q, state):
        if np.random.random() <= self.epsilon:
            return np.random.randint(0, len(q[state]) - 1)

        m = np.amax(q[state])
        indices = np.nonzero(q[state] == m)[0]
        return np.random.choice(indices)

    def train_hdqn(self, batch_inputs):
        states, actions, next_states, reward = [i for i in zip(*batch_inputs)]

        # get target q-values and predicted Q-values
        target_q_values = []
        for i, state in enumerate(states):
            current_q_values = self.meta_controller.get_q_values(state)
            action_index = self.meta_controller_agent.get_index(actions[i])
            current_q_values[action_index] = reward[i] + (
                    self.discount_factor * max(self.target_meta_controller.get_q_values(next_states[i])))
            target_q_values.append(current_q_values)
        target_q_values = autograd.Variable(torch.from_numpy(np.array(target_q_values)).type(dtype)).type(dtype)

        # traind
        self.meta_controller.optimizer.zero_grad()
        reshaped_states = [autograd.Variable(torch.from_numpy(i.flatten()).type(dtype)).data.numpy() for i in states]
        _reshaped_states = autograd.Variable(torch.from_numpy(np.array(reshaped_states)).type(dtype))

        outputs = self.meta_controller(_reshaped_states)
        self.target_meta_controller.load_state_dict(self.meta_controller.state_dict())
        loss = self.meta_controller.criterion(outputs, target_q_values)
        loss.backward()
        self.meta_controller.optimizer.step()


class DQN(nn.Module):

    def __init__(self, in_features, out_features, learning_rate):
        """
        Initialize a Meta-Controller of Hierarchical DQN network for the diecreate mdp experiment
            in_features: number of features of input.
            out_features: number of features of output.
                Ex: goal for meta-controller or action for controller
        """
        super(DQN, self).__init__()
        self.in_features = in_features
        # self.conv1 = nn.Conv2d(in_features, 18, kernel_size=3, stride=1, padding=1)
        # self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc = nn.Linear(in_features=in_features, out_features=260)
        self.fc1 = nn.Linear(260, out_features)

        self.criterion = F.smooth_l1_loss
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)

    def forward(self, input):
        # x = F.relu(self.conv1(input))
        # x = self.pool(x)
        # x = x.view(-1, 18 * 10 * 10)
        x = F.relu(self.fc(input))
        x = self.fc1(x)
        return x

    def get_q_values(self, state):
        state = autograd.Variable(torch.from_numpy(state).type(dtype))
        return self.forward(state.flatten()).data.numpy()


class Qlearning(object):

    def __init__(self, agent_maze):
        self.maze = agent_maze
        self.q = defaultdict(lambda: np.zeros(len(self.maze.actions)))
