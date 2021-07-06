from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

import matplotlib

max_grid_length = 4
starting_grid_pos = 0
episode_length = 500


# Env inherited from Open AI Gym
class Grid(Env):

    def __init__(self):

        # Actions: up, down, left, right
        self.action_space = Discrete(4)
        # grid cells
        self.observation_space = Box(low=np.array([starting_grid_pos, starting_grid_pos]),
                                     high=np.array([max_grid_length, max_grid_length]), dtype=int)

        # start position
        self.state = np.array([starting_grid_pos, starting_grid_pos])

        # starting episode length
        self.path_length = 1

        self.wall = False

        self.success = False

        self.target_state = np.array([max_grid_length, max_grid_length])

    def step(self, action):
        # action| path_length | state
        # 0     |  +1         | (_, +1)
        # 1     |  +1         | (_, -1)
        # 2     |  +1         | (-1, _)
        # 3     |  +1         | (+1, _)
        # self.state è nella forma [x,y]

        # Calcolo distanza euclidea del "vecchio stato":
        old_target_distance = np.linalg.norm(self.state - self.target_state)

        if action == 0:  # GO UP
            self.state[1] += 1
            if (state[1] <= max_grid_length):
                # Il movimento resta all'interno della griglia
                pass
            else:  # Il movimento esce dai limiti della griglia:
                #print("mossa in alto non valida!")
                # Ritorno allo stato precedente
                self.state[1] -= 1
                env.action_space = np.array([1, 2, 3])
                action = random.choice(env.action_space)
                env.step(action)

        if action == 1:  # GO DOWN
            self.state[1] -= 1
            if (state[1] >= starting_grid_pos):
                # Il movimento resta all'interno della griglia
                pass
            else:
                # Il movimento esce dai limiti della griglia:
                #print("Mossa in basso non valida!")
                self.state[1] += 1  # Ritorno allo stato precedente
                env.action_space = np.array([0, 2, 3])
                action = random.choice(env.action_space)
                env.step(action)

        if action == 2:  # GO LEFT
            self.state[0] -= 1
            if (state[0] >= starting_grid_pos):
                # Il movimento resta all'interno della griglia
                pass
            else:
                #print("Mossa a sinistra non valida!")
                self.state[0] += 1
                env.action_space = np.array([0, 1, 3])
                action = random.choice(env.action_space)
                env.step(action)

        if action == 3:  # GO RIGHT
            self.state[0] += 1
            if (state[0] <= max_grid_length):
                # Il movimento resta all'interno della griglia
                pass
            else:
                #print("Mossa a destra non valida!")
                self.state[0] -= 1
                env.action_space = np.array([0, 1, 2])
                action = random.choice(env.action_space)
                env.step(action)

        # Grow path length by 1
        self.path_length += 1

        # Calculate the euclidian distance from the current stat to the target state
        # Calcolo la distanza euclidea del nuovo stato
        new_target_distance = np.linalg.norm(self.state - self.target_state)

        # Calculate reward
        if self.state[0] == self.target_state[0] and self.state[1] == self.target_state[1]:
            reward = 100
            print("TARGET STATE RAGGIUNTO, termine episodio")
        elif new_target_distance <= old_target_distance:
            reward = -1
        else:
            reward = 0

        # Check if search is done
        if self.path_length > episode_length:
            done = True
            print ("Fine episodio causa step terminati")
        elif self.state[0] == self.target_state[0] and self.state[1] == self.target_state[1]:
            # Se arrivo nello stato target allora l'episodio termina
            done = True
            self.success= True
            print("Fine episodio causa target state raggiunto")
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
        self.path_length = 1
        return self.state


########## END GRID CLASS ##################

# Crea un'istanza della classe Grid
env = Grid()

episodes = 30


# Learning rate: how much you accept the new value vs the old value.
lr = 0.8

# Gamma: discount factor
gamma = 0.9


# Aggiornamento dei Q-values nella matrice Q_table
def update_qtable(state, action, reward, new_state):
    # Cumulative reward
    # DA RIVEDERE VALORE DI LR (aka learning rate)
    Q_table[state, action] += lr * (reward + gamma * np.max(Q_table[new_state]) - Q_table[state, action])

    return Q_table


# Mappa gli stati della matrice dell'enviorment in un vettore lungo 25, come i tutti i possibili stati
def mapping_state(next_state):
    # [0,0] -> 0; [0,1] -> 1 ... [1,0] -> 5

    new_pos = (next_state[1] * 5) + (next_state[0] * 1)

    return new_pos


for episode in range(episodes):
    print("######### NEW EPISODE ################")
    state = env.reset()
    done = False
    score = 0

    # Resett Q_table ad ogni episodio
    Q_table = np.zeros(shape=(25, 4), dtype=float)
    while not done:
        env.render()
        # Ripristino il vettore delle azioni in modo da poterle eseguire tutte di nuovo
        env.action_space = np.array([0, 1, 2, 3])

        epsilon = 1 / env.path_length
        not_epislon = 1 - (1 / env.path_length)

        if random.uniform(0, 1) < epsilon:
            """ Explore: select a random action   """
            # Choose action
            action = random.choice(env.action_space)
            print("Exploration action: ", action)
            state2 = mapping_state(state)
            # Perform action
            next_state, reward, done, info = env.step(action)

            mapped_state = mapping_state(next_state)
            # Aggiorna la Q_table
            update_qtable(state2, action, reward, mapped_state)
            score += reward
        else:
            """ Exploit: select the action with max value (future reward)  """
            state2 = mapping_state(state)
            action_qvalues = Q_table[state2].tolist()
            max_action_value = max(action_qvalues)
            #Choose action
            action = action_qvalues.index(max_action_value)
            print ("Exploit action: ", action)
            #Perform action
            next_state, reward, done, info = env.step(action)
            mapped_state = mapping_state(next_state)
            # Aggiorna la Q_table
            update_qtable(state2, action, reward, mapped_state)
            score += reward


    print('EPISODE:{} SCORE:{} STEP:{}, SUCCESS:{} \nFINAL Q_table:{}'.format(episode, score, env.path_length,env.success,  Q_table))

print("Fine programma")
