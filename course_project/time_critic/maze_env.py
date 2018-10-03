import numpy as np


class Maze(object):

    def __init__(self, maze, rat=(0, 0), target=None, name=""):
        self.name = name

        self.LEFT = 0
        self.RIGHT = 1
        self.UP = 2
        self.DOWN = 3
        self.rat_init = rat
        self.rat_mark = 0.5  # The current rat cell will be painteg by gray 0.5

        # Actions dictionary
        self.actions = [self.LEFT, self.RIGHT, self.UP, self.DOWN]

        self.num_actions = len(self.actions)

        self._maze = np.array(maze)
        nrows, ncols = self._maze.shape
        self.target = target if target else (nrows - 1, ncols - 1)  # target cell where the "cheese" is
        self.free_cells = [(r, c) for r in range(nrows) for c in range(ncols) if self._maze[r, c] == 1.0]
        self.free_cells.remove(self.target)
        if self._maze[self.target] == 0.0:
            raise Exception("Invalid maze: target cell cannot be blocked!")
        if rat not in self.free_cells:
            raise Exception("Invalid Rat Location: must sit on a free cell")
        self.reset(rat)

    def reset(self, rat):
        self.rat = rat
        self.maze = np.copy(self._maze)
        row, col = rat
        self.maze[row, col] = self.rat_mark
        self.state = (row, col, 'start')
        self.total_reward = 0
        self.visited = set()

    def update_state(self, action):
        nrows, ncols = self.maze.shape
        nrow, ncol, nmode = rat_row, rat_col, mode = self.state

        if self.maze[rat_row, rat_col] > 0.0:
            self.visited.add((rat_row, rat_col))  # mark visited cell

        valid_actions = self.valid_actions()

        if not valid_actions:
            nmode = 'blocked'
        elif action in valid_actions:
            nmode = 'valid'
            if action == self.LEFT:
                ncol -= 1
            elif action == self.UP:
                nrow -= 1
            if action == self.RIGHT:
                ncol += 1
            elif action == self.DOWN:
                nrow += 1
        else:  # invalid action, no change in rat position
            nmode = 'invalid'

        # new state
        self.state = (nrow, ncol, nmode)

    def get_reward(self):
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows - 1 and rat_col == ncols - 1:
            return 1.0
        if mode == 'blocked':
            return -1.0
        if (rat_row, rat_col) in self.visited:
            return -0.25
        if mode == 'invalid':  # obstacle
            return -0.75
        if mode == 'valid':
            return -0.04

    def act(self, action):
        self.update_state(action)
        reward = self.get_reward()
        self.total_reward += reward
        status = self.game_status()
        envstate = self.observe()
        return envstate, reward, status

    def observe(self):
        canvas = self.draw_env()
        envstate = canvas.reshape((1, -1))
        return envstate

    def draw_env(self):
        canvas = np.copy(self.maze)
        nrows, ncols = self.maze.shape
        # clear all visual marks
        for r in range(nrows):
            for c in range(ncols):
                if canvas[r, c] > 0.0:
                    canvas[r, c] = 1.0
        # draw the rat
        row, col, valid = self.state
        canvas[row, col] = self.rat_mark
        return canvas

    def game_status(self):
        # if self.total_reward < self.min_reward:
        #     return 'lose'
        rat_row, rat_col, mode = self.state
        nrows, ncols = self.maze.shape
        if rat_row == nrows - 1 and rat_col == ncols - 1:
            return 'win'
        return 'not_over'

    def valid_actions(self, cell=None):
        # valid when the agent dont move into the obstacle
        if cell is None:
            row, col, mode = self.state
        else:
            row, col = cell
        actions = [self.LEFT, self.UP, self.RIGHT, self.DOWN]
        nrows, ncols = self.maze.shape
        if row == 0:
            actions.remove(self.UP)
        elif row == nrows - 1:
            actions.remove(self.DOWN)

        if col == 0:
            actions.remove(self.LEFT)
        elif col == ncols - 1:
            actions.remove(self.RIGHT)

        if row > 0 and self.maze[row - 1, col] == 0.0:
            actions.remove(self.UP)
        if row < nrows - 1 and self.maze[row + 1, col] == 0.0:
            actions.remove(self.DOWN)

        if col > 0 and self.maze[row, col - 1] == 0.0:
            actions.remove(self.LEFT)
        if col < ncols - 1 and self.maze[row, col + 1] == 0.0:
            actions.remove(self.RIGHT)

        return actions
