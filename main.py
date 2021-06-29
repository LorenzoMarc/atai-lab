from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import nnModel
from tensorflow.keras.optimizers import Adam

# Env inherited from Open AI Gym
class Grid(Env):

    def __init__(self):
        # Actions: up, down, left, right
        self.action_space = Discrete(4)
        # grid cells
        self.observation_space = Box(low=np.array([0, 0]), high=np.array([5, 5]), dtype=int)

        # start position
        self.state = np.array([0, 0])

        # episode length
        self.path_length = 10

        self.wall = False

        self.target_state = np.array([5, 5])

    def step(self, action):
        # action| path_length | state
        # 0     |  -1         | (_, +1)
        # 1     |  -1         | (_, -1)
        # 2     |  -1         | (-1, _)
        # 3     |  -1         | (+1, _)
        if action == 0:
            self.state[1] += 1
            print(self.state)
        if action == 1:
            self.state[1] -= 1
            print(self.state)
        if action == 2:
            self.state[0] -= 1
            print(self.state)
        if action == 3:
            self.state[0] += 1
            print(self.state)

        # Reduce path length by 1
        self.path_length -= 1


        # Check this distance. Choose appropriate distance measure
        target_distance = np.linalg.norm(self.state - self.target_state)

        # Calculate reward
        if target_distance <= 2:
            reward = 1
        else:
            reward = -1

            # Check if search is done
        if self.path_length <= 0:
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
        self.state = np.array([0, 0])
        # Reset path
        self.path_length = 10
        return self.state

env = Grid()

episodes = 10

for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        while n_state[0] < 0 or n_state[1] < 0:
            env.reset()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
        score += reward
    print('episode:{} Score:{}'.format(episode,score))

states = env.observation_space.shape
print(states)
actions = env.action_space.n
print(actions)

model = nnModel.build_model(states, actions)

dqn = nnModel.build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

