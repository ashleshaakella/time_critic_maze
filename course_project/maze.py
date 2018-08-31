from __future__ import print_function

from reinforcement_algorithms import RL
from maze_matrix import MAZE_MATRIX
import numpy as np
from maze_env import Qmaze


def play_game(model, qmaze, rat_cell):
    qmaze.reset(rat_cell)
    envstate = qmaze.observe()
    while True:
        prev_envstate = envstate
        # get next action
        q = model.predict(prev_envstate)
        action = np.argmax(q[0])
        # apply action, get rewards and new state
        envstate, reward, game_status = qmaze.act(action)
        if game_status == 'win':
            return True
        elif game_status == 'lose':
            return False


def completion_check(model, qmaze):
    for cell in qmaze.free_cells:
        if not qmaze.valid_actions(cell):
            return False
        if not play_game(model, qmaze, cell):
            return False
    return True


if __name__=="__main__":
    maze = MAZE_MATRIX[1]
    agent_number = 0
    qmaze = Qmaze(maze)
    algo = RL(agent=qmaze, number_of_episodes=1000, discount_factor=1, learning_rate=0.1, max_tries=20000, epsilon=0.1)
    # algo.expected_sarsa()
    algo.q_learning()
    algo.play_q_learning()
    pass


