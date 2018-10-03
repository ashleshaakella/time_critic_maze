from time_critic_maze_env import TimeCriticMazeEnv
from HDQL import HierarchicalSemiDeepQLearning
import pickle


if __name__ == "__main__":
    multi_agent_maze = TimeCriticMazeEnv(number_of_agents=4)
    hdqn = HierarchicalSemiDeepQLearning(maze=multi_agent_maze,
                                         discount_factor=0.1,
                                         number_of_episodes=2500,
                                         learning_rate=0.1,
                                         max_tries=2000,
                                         epsilon=0.5)
    hdqn.q_learning()
    with open('hdqn_multi_agent_mice.pik', 'w') as io:
        pickle.dump(hdqn)