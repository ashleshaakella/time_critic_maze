##### current time critic maze (problem 1)
* agents are initialized in the same position in all episodes
* agents have common obstacles
* one goal
* hierarchical agents with one met-controller and 3 controllers
* collision avoidance

##### MPD for problem 1 
###### Meta controller
* State space is n^n where n is the number of agents
* action space is n^n where n is the number of agents
* meta controller have a pre-processor which build the state
* reward:
    ```
    (number_of_agents_collided * -1) + (number_of_agents_reached_goal * 0.5) - 0.5
    ```
* Meta controller is Deep Q-network with neural network of 3 layers. Input layer is of sixe of 400 (20*20 grid size)
* output layer of DQN contain 16 nodes. Which is equal to (2 ^ number of controllers)

###### Controller
* State space of controller is x*y where x and y are gird size
* action space is 4 (up, right, down, left)
* rewards
    ```
    * If it reach goal = 1.0
    * if it is bloacked = -1
    * revisiting the same cell = -0.25
    * hitting an obstacle = -0.25
    * make a correct move = -0.04 (for every step agent takes)
    ```
###### analysis 
* save the dictionary of the following
    * number of time controller updated a state
    * Q values of each agent (meta controller and controller)
    * reward average over all the episodes (both controller and meta-controller)
    * save the weights of (DQN) and q values of the controllers.

##### Future modifications
* Heterogeneous agents
* multiple goals
* All agents reaching goal on time (or) reaching in a certain order
* Hierarchical RL where meta controller agnet can control the speed of controller agents
