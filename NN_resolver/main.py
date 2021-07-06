from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import nnModel
import random
import math
from tensorflow.keras.optimizers import Adam
import time
from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt


class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

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

    def __init__(self, walls_array, max_dim, path_length):
        # Actions: up, down, left, right
        self.action_space = np.array([0, 1, 2, 3])
        # grid cells as flatten array
        self.observation_space = Box(low=np.array([0]), high=np.array([max_dim]), dtype=int)
        #self.observation_space = proviamo.setarray('mazes/generated_maze_1.png')
        # start position
        self.state = np.array([0])

        # episode length
        self.path_length = path_length
        # agent2 or maze's exit
        self.target_state = np.array([max_dim])

        # subset of walls
        self.walls_states = walls_array

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
            reward = 1500
            # print("TARGET HIT!")
        # decrease reward if a wall is hit
        elif self.state in self.walls_states:
            reward = -2000
            # print("WALL HIT!")
        # decrease reward if negative state(out of env)
        elif self.state < 0 or self.state > max_dim:
            reward = -2000
            # print("OUT OF BOUNDS")
        else:
            reward = 0
            # print("AVAILABLE PATH")

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
        self.path_length =500
        return self.state

walls = [1, 9, 17, 25, 5, 13, 21, 60, 52, 44, 58]
max_dim = 64
path_length = 500
#require 'walls' array: the states where walls are
# require maximum state ( e.g.: a matrix 5x5 has 0..24 states, so 24 is max)
# require path length --> number of steps available to the agent per episode
env = Grid(walls, max_dim, path_length)

'''
# Random walk on maze in 100 episodes
episodes = 20

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
    print('episode:{} Score:{}'.format(episode, score))
'''
# Input for NN model and agent
states = env.observation_space.shape
actions = len(env.action_space)
# Dense Net is build with states and actions
model = nnModel.build_model(states, actions)

model.summary()

# Deep Q agent is built, compiled with the network
# and fitted in the env. It have been trained for 50000 steps.
# Test scores are calculated in 100 episodes
time_callback = TimeHistory()
dqn = nnModel.build_agent(model, actions)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
scores = dqn.fit(env, nb_steps=50000, visualize=False, verbose=2,
                 callbacks=[time_callback])

print(scores.history['episode_reward'])

print(time_callback.times)


plt.plot(time_callback.times)
# naming the x and y axis
plt.xlabel('Number of Episodes')
plt.ylabel('Computation time per episode (sec)')

# plotting a line plot after changing it's width and height
f = plt.figure()
f.set_figwidth(16)
f.set_figheight(9)
plt.show()

plt.hist(time_callback.times)
plt.show()

scores = dqn.test(env, nb_episodes=10, visualize=False, callbacks=[time_callback])
print(np.mean(scores.history['episode_reward']))

print(time_callback.times)

plt.plot(time_callback.times)
# naming the x and y axis
plt.xlabel('Number of Episodes')
plt.ylabel('Computation time per episode (sec)')

# plotting a line plot after changing it's width and height
f = plt.figure()
f.set_figwidth(16)
f.set_figheight(9)
plt.show()
dqn.save_weights('dqn_weights.h5f', overwrite=True)