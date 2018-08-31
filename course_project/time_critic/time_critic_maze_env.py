from maze_env import Qmaze


class TimeCriticMazeEnv(Qmaze):
    def __init__(self, number_of_agents=3, number_of_goals=1):
        # builds a maze with the following
        #      - obstacles (where agents cant go through)
        #      - hidden obstacles (where agent can go but not visible to the other agents)
        #      - will have randomized number_of_goals
        #      - agents with randomized starting position
        #           Total number of agents = number_of_agents
        #           Agents starting position will be a position on gird other that obstacles or hidden obstacles.
        # Have following functionality
        #      - reset (env reset to initial maze)
        #      -
        pass
