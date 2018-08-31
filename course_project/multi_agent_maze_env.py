import numpy as np
from reinforcement_algorithms import RL
from collections import defaultdict
from maze import Qmaze
import matplotlib.pyplot as plt
from agent_color_map import AGENT_COLOR


class MultiAgentMaze(Qmaze):

    def __init__(self, number_of_agents, maze):
        self.maze = maze
        self.numer_of_agents = number_of_agents

        self.mouse_start_position = self._init_start_position()
        self.agents = [Qmaze(maze,
                             rat=self.mouse_start_position[i],
                             name="mouse: " + str(i),
                             init_canvas=False)
                       for i in range(number_of_agents)]
        self.agent_color_pref = next(filter(lambda x:x.get("MULTI_AGENT", False), AGENT_COLOR))
        self._init_canvas()

    def _init_start_position(self):
        nrows, ncols = self.maze.shape
        target = (nrows - 1, ncols - 1)  # target cell where the "cheese" is
        free_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self.maze[r, c] == 1.0]
        free_cells.remove(target)

        agent_start_positions = []
        for i in range(self.numer_of_agents):
            not_valid = True
            while not_valid:
                pos = (np.random.choice(nrows), np.random.choice(ncols))
                if pos not in agent_start_positions and pos in free_cells:
                    not_valid=False

                if not not_valid:
                    agent_start_positions.append(pos)
        return agent_start_positions

    def show(self):
        nrows, ncols = self.maze.shape
        canvas = np.copy(self.maze)
        canvas[canvas == 0] = self.agent_color_pref['OBSTACLES_COLOR']
        canvas[canvas == 1] = self.agent_color_pref['GRID_COLOR']
        canvas[nrows - 1, ncols - 1] = self.agent_color_pref['CHEESE_COLOR']

        for i, agent in enumerate(self.agents):
            for row, col in agent.visited:
                canvas[row, col] = self.agent_color_pref['VISITED_COLOR'][i]
            rat_row, rat_col, _ = agent.state
            canvas[agent.rat_init[0], agent.rat_init[1]] = self.agent_color_pref['START_COLOR'][i]

        for i, agent in enumerate(self.agents):
            rat_row, rat_col, _ = agent.state
            canvas[rat_row, rat_col] = self.agent_color_pref['MOUSE_COLOR'][i]  # rat cell

        self.img.set_data(canvas)
        plt.show()

    def _init_canvas(self, show_start_color=False):
        self.fig = plt.figure()
        self.fig.legend
        plt.grid('on')
        nrows, ncols = self.maze.shape
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        canvas = np.copy(self.maze)

        canvas[canvas == 0] = self.agent_color_pref['OBSTACLES_COLOR']
        canvas[canvas == 1] = self.agent_color_pref['GRID_COLOR']
        for i, agent in enumerate(self.agents):
            for row, col in agent.visited:
                canvas[row, col] = self.agent_color_pref['VISITED_COLOR'][i]

            rat_row, rat_col, _ = agent.state
            canvas[rat_row, rat_col] = self.agent_color_pref['MOUSE_COLOR'][i]  # rat cell
            if show_start_color:
                canvas[agent.rat_init[0], agent.rat_init[1]] = self.agent_color_pref['START_COLOR']

        canvas[nrows - 1, ncols - 1] = self.agent_color_pref['CHEESE_COLOR']  # cheese cell
        self.img = plt.imshow(canvas, interpolation='none', cmap=plt.cm.get_cmap(self.agent_color_pref["CMAP"], 40),
                              animated=True)
        plt.show()


class MultmiAgentRL(RL):

    def __init__(self, multi_agent, number_of_episodes, discount_factor, learning_rate, max_tries, epsilon=0.1):
        self.mult_agent = multi_agent
        self.agents = self.mult_agent.agents
        self.number_of_episodes = number_of_episodes
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.max_tries = max_tries
        self.epsilon = epsilon
        self.q_for_each_agent = [defaultdict(lambda: np.zeros(agent.num_actions)) for agent in self.agents]

    def q_learning(self):
        rewards_per_agent = defaultdict(lambda: [])
        for i in range(self.number_of_episodes):
            # reset agents
            [agent.reset(agent.rat_init) for agent in self.agents]
            _reward = self.run_episode()
            for agent_num, reward in _reward.items():
                rewards_per_agent[agent_num] += reward
            if i%10 == 0 and i!= 0:
                for agent_num, rewards in rewards_per_agent.items():
                    print("Max reward for {agent} agent is {max_reward}".format(agent=agent_num,
                                                                                max_reward=max(rewards[i-10:i])))

    def run_episode(self):
        rewards = defaultdict(lambda: [])
        agent_reached_goal = [False for i in self.agents]
        timesteps = [0 for i in self.agents]
        states = list(map(lambda x:(x[0], x[1]), map(lambda x:x.state, self.agents)))
        while True:
            if all([agent_reached_goal[i] for i in range(len(self.agents))]):
                break
            for i, agent in enumerate(self.agents):
                if not agent_reached_goal[i]:
                    # play only if the agent didnt reach goal
                    action = self.get_argmax_action(self.q_for_each_agent[i], state=states[i])
                    env_state, reward, status = agent.act(action)
                    new_state = (agent.state[0], agent.state[1])
                    self.q_for_each_agent[i][states[i]][action] += self.learning_rate * (
                            reward + (self.discount_factor * max(self.q_for_each_agent[i][new_state])) - self.q_for_each_agent[i][states[i]][action])
                    timesteps[i] += 1
                    states[i]= new_state
                    if status == 'win' or timesteps[i] >= self.max_tries:
                        rewards[i].append(agent.total_reward)
                        self.mult_agent.show()
                        agent_reached_goal[i] = True
            if any([k%10 == 0 for k in timesteps]):
                self.mult_agent.show()

        return rewards

    def play_q_learning(self):
        [agent.reset(agent.rat_init) for agent in self.agents]
        agent_reached_goal = [False for i in self.agents]
        timesteps = [0 for i in self.agents]
        while True:
            if all([agent_reached_goal[i] for i in range(len(self.agents))]):
                break
            for i, agent in enumerate(self.agents):
                if not agent_reached_goal[i]:
                    (row, col, mode) = agent.state
                    action = self.get_argmax_action(self.q_for_each_agent[i], (row, col))
                    env_state, reward, status = agent.act(action)
                    timesteps[i] += 1
                    if status == 'win':
                        agent_reached_goal[i] = True
            self.mult_agent.show()
        for agent_num, ts in enumerate(timesteps):
            print("Agent {agent} reached goal in {ts}".format(agent=agent_num, ts=ts))
        return


class MultiAgentCollisionControlRL(MultmiAgentRL):

    def __init__(self):
        pass
