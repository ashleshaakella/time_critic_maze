import numpy as np
import itertools
from itertools import groupby
from copy import deepcopy
import matplotlib.pyplot as plt
from maze_env import Maze
from maze_matrix import MAZE_MATRIX
from agent_color_map import AGENT_COLOR


class TimeCriticMazeEnv(Maze):

    def __init__(self, number_of_agents=3, maze_matrix=MAZE_MATRIX[0]):
        # builds a maze with the following
        #      - obstacles (where agents cant go through)
        #      - agents with randomized starting position
        #           Total number of agents = number_of_agents
        #           Agents starting position will be a position on gird other that obstacles or hidden obstacles.

        self.maze_matrix = maze_matrix
        self.number_of_agents = number_of_agents

        self.mouse_start_position = self._init_start_position()
        self.agents = [Maze(self.maze_matrix,
                            rat=self.mouse_start_position[i],
                            name="mouse: " + str(i)) for i in range(number_of_agents)]
        self.meta_controller = MetaControllerAgent(controller_agents=self.agents, env=self.maze_matrix)
        self.agent_color_pref = next(
            filter(lambda x: x.get("MULTI_AGENT", False) and len(x["MOUSE_COLOR"]) == len(self.agents), AGENT_COLOR))
        if not self.agent_color_pref:
            raise Exception('Agent color preference is not defined correctly!')
        self._init_canvas()

    def _init_start_position(self):
        nrows, ncols = self.maze_matrix.shape
        target = (nrows - 1, ncols - 1)  # target cell where the "cheese" is
        free_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self.maze_matrix[r, c] == 1.0]
        free_cells.remove(target)

        agent_start_positions = [(1,1), (2,2), (1,2), (2,1)]
        return agent_start_positions
        # for i in range(self.number_of_agents):
        #     not_valid = True
        #     while not_valid:
        #         pos = (np.random.choice(nrows), np.random.choice(ncols))
        #         if pos not in agent_start_positions and pos in free_cells:
        #             not_valid = False
        #
        #         if not not_valid:
        #             agent_start_positions.append(pos)
        # return agent_start_positions

    def show(self):
        nrows, ncols = self.maze_matrix.shape
        canvas = np.copy(self.maze_matrix)
        canvas[canvas == 0] = self.agent_color_pref['OBSTACLES_COLOR']
        canvas[canvas == 1] = self.agent_color_pref['GRID_COLOR']
        canvas[nrows - 1, ncols - 1] = self.agent_color_pref['CHEESE_COLOR']

        # for i, agent in enumerate(self.agents):
        #     # for row, col in agent.visited:
        #     #     canvas[row, col] = self.agent_color_pref['VISITED_COLOR'][i]
        #     rat_row, rat_col, _ = agent.state
        #     canvas[agent.rat_init[0], agent.rat_init[1]] = self.agent_color_pref['START_COLOR'][i]

        for i, agent in enumerate(self.agents):
            rat_row, rat_col, _ = agent.state
            canvas[rat_row, rat_col] = self.agent_color_pref['MOUSE_COLOR'][i]  # rat cell

        self.img.set_data(canvas)
        self.fig.canvas.draw()
        return self.img

    def _init_canvas(self, show_start_color=False):
        plt.ion()
        self.fig = plt.figure()
        ax = self.fig.add_subplot(111)
        plt.grid('on')
        nrows, ncols = self.maze_matrix.shape
        ax = plt.gca()
        ax.set_xticks(np.arange(0.5, nrows, 1))
        ax.set_yticks(np.arange(0.5, ncols, 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        canvas = np.copy(self.maze_matrix)

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


class MetaControllerAgent(object):

    def __init__(self, controller_agents, env):
        self.controller_agents = controller_agents

        self.controller_values = [i+3 for i in range(len(self.controller_agents))]
        self.state = deepcopy(env)
        for i, agent in enumerate(self.controller_agents):
            self.state[agent.rat[0]][agent.rat[1]] = self.controller_values[i]

        self.target = next(map(lambda x:x.target, self.controller_agents))
        # actions for meta controller
        self.actions = list(itertools.product([0, 1], repeat=len(self.controller_agents)))
        self.actions.pop(0)
        # self.actions = [(1,1), (1,0), (0,1)]

    def get_action(self, index):
        return self.actions[index]

    @property
    def random_action_index(self):
        return np.random.randint(0, len(self.actions)-1)

    def get_index(self, action):
        return self.actions.index(tuple(action))

    def update_state(self, controller_agent_position):
        for i, (x,y) in enumerate(controller_agent_position):
            self.state[x][y] = self.controller_values[i]

    @property
    def current_state(self):
        return self.state

    def collision(self, agents_positions):
        agents_didnt_reach_goal = list(filter(lambda x:x != self.target, agents_positions))
        num_of_agents_collided = {}
        for i in agents_didnt_reach_goal:
            num_of_agents_collided.setdefault(i, 0)
            num_of_agents_collided[i] +=1

        if list(filter(lambda x:x >1, num_of_agents_collided.values())):
            return True
        return False

    def reward(self, agents_positions):
        # reward function
        # reward = ((number of agent collided * -1) + (number of agents reached goal * 0.5) - 0.5)
        # the first part is to penalize the meta controller for collision
        # the second part is to give positive reward for agents reaching goal
        # the third part is to bias
        # number of agents collides
        agents_didnt_reach_goal = list(filter(lambda x: x != self.target, agents_positions))
        agents_collided = {}
        for i in agents_didnt_reach_goal:
            agents_collided.setdefault(i, 0)
            agents_collided[i] += 1
        num_of_agents_collided = 0
        for num_agent in agents_collided.values():
            if num_agent >1:
                num_of_agents_collided +=num_agent

        # number of agents reached goal
        num_of_agents_reached_target = len(list(filter(lambda x:x == self.target, agents_positions)))
        return (num_of_agents_collided*-1) + (num_of_agents_reached_target*0.5) - 0.5

