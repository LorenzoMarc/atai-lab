from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import nnModel
import random
import math

from tensorflow.keras.optimizers import Adam

#########################################################################
'''
TASK: 

Train a NN to resolve a maze resolution with Q Agent.
The baseline environment is inherited from OpenAI Gym.
Elements:
- Custom Grid: to complete the environment
- Step function: to proceed on training 
- render function (not used)
- reset function: bring state to the starting position
'''

# Env inherited from Open AI Gym
# Grid is a representation of 5x5 dimension
class Grid(Env):

    def __init__(self):
        # Actions: up, down, left, right
        self.action_space = np.array([0, 1, 2, 3])
        # grid cells as flatten array
        self.observation_space = Box(low=np.array([0]), high=np.array([24]), dtype=int)

        # start position
        self.state = np.array([0])

        # episode length
        self.path_length = 1000
        # agent2 or maze's exit
        self.target_state = np.array([24])

        # subset of walls
        self.walls_states = [1, 6, 11, 13, 18, 23]

    def step(self, action):
        # action| path_length | state
        # 0     |  -1         | (_, +1)
        # 1     |  -1         | (_, -1)
        # 2     |  -1         | (-1, _)
        # 3     |  -1         | (+1, _)
        if action == 0:
            self.state -= 5

        if action == 1:
            self.state += 5

        if action == 2:
            self.state -= 1

        if action == 3:
            self.state += 1

        # Reduce path length by 1
        self.path_length -= 1

        # Calculate reward if target state is reached
        if self.state == self.target_state:
            reward = 500
        # decrease reward if a wall is hit
        elif [self.state == x for x in self.walls_states]:
            reward = -1
        # decrease reward if negative state(out of env)
        elif self.state < 0:
            reward = -1
        else:
            reward = 0

        # Check if episode is done
        if self.path_length <= 0 or self.state == self.target_state:
            done = True
        else:
            done = False

        # Set placeholder for info
        info = {}

        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass

    def reset(self):
        # Reset path temperature
        self.state = np.array(0)
        # Reset path
        self.path_length = 10
        return self.state


env = Grid()

'''
# Random walk on maze in 100 episodes
episodes = 100

for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        if state in [1, 2, 3]:
            env.action_space = np.array([1 ,2 ,3])
        elif state in [5, 10, 15]:
            env.action_space = np.array([0,1,3])
        elif state in [9, 14, 19]:
            env.action_space = np.array([0,1,2])
        elif state in [21, 22, 23]:
            env.action_space = np.array([0,2,3])
        elif state == 0:
            env.action_space = np.array([1,3])
        elif state == 4:
            env.action_space = np.array([1,2])
        elif state == 20:
            env.action_space = np.array([0,3])
        elif state == 24:
            env.action_space = np.array([0,2])
        else:
            env.action_space = np.array([0,1,2,3])
        action = random.choice(env.action_space)
        n_state, reward, done, info = env.step(action)
        score += reward
    #print('episode:{} Score:{}'.format(episode, score))
'''
# Input for NN model and agent
states = env.observation_space.shape
actions = len(env.action_space)
# Dense Net is build with status
model = nnModel.build_model(states, actions)
model.summary()

# Deep Q agent is built, compiled with the network
# and fitted in the env. It have been trained for 50000 steps.
# Test scores are calculated in 100 episodes
dqn = nnModel.build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=5000, visualize=False, verbose=1)
scores = dqn.test(env, nb_episodes=100, visualize=False)
print(np.mean(scores.history['episode_reward']))