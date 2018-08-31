from maze_matrix import MAZE_MATRIX
from multi_agent_maze_env import MultiAgentMaze
from multi_agent_maze_env import MultiAgentRL

if __name__=="__main__":
    number_of_agents = 2
    maze = MAZE_MATRIX[1]
    multi_agent_maze = MultiAgentMaze(maze=maze, number_of_agents=number_of_agents)
    algo = MultiAgentRL(multi_agent=multi_agent_maze,
                        number_of_episodes=1000,
                        discount_factor=1,
                        learning_rate=0.1,
                        max_tries=2000)
    algo.q_learning()
    algo.play_q_learning()
    pass
