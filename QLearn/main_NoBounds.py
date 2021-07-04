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


    def oracle(self, actual_state, action):
        #print ("oracolo in esecuzione!")

        if action == 0 and state[1] < max_grid_length:  # GO UP
            #actual_state[1] += 1
            return True

        elif action == 1 and state[1] > starting_grid_pos:  # GO DOWN
            #actual_state[1] -= 1
            return True

        elif action == 2 and state[0] > starting_grid_pos:  # GO LEFT
            #actual_state[0] -= 1
            return True

        elif action == 3 and state[0] < max_grid_length:  # GO RIGHT
            #actual_state[0] += 1
            return True

        else:
            print("Non valid action!")
            return False

        '''
        next_state = mapping_state(actual_state)
        if 0 < next_state or next_state > 24: #MAGIC NUMBERS
            print ("next_state: ", next_state)
            return True, next_state
        else:
            print("next_state: ", next_state)
            return False, next_state
        '''


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

        if action == 1:  # GO DOWN
            self.state[1] -= 1

        if action == 2:  # GO LEFT
            self.state[0] -= 1

        if action == 3:  # GO RIGHT
            self.state[0] += 1

        # Grow path length by 1
        self.path_length += 1

        # Calculate the euclidian distance from the current stat to the target state
        # Calcolo la distanza euclidea del nuovo stato
        new_target_distance = np.linalg.norm(self.state - self.target_state)

        # Calculate reward
        if self.state[0] == self.target_state[0] and self.state[1] == self.target_state[1]:
            reward = 10000
            print("TARGET STATE RAGGIUNTO, termine episodio")

        elif new_target_distance < old_target_distance:
            reward = +1
        else:
            reward = -1

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
        print("self.state: ", self.state, " target_state: ", self.target_state, " new distance: ", new_target_distance,
              " old distance: ", old_target_distance)
        # Return step information
        return self.state, reward, done, info


    def render(self):
        # Implement viz
        pass

    def reset(self):
        # Reset path temperature
        self.state = np.array([starting_grid_pos, starting_grid_pos])
        # Reset path
        self.path_length = 1
        return self.state








########## END GRID CLASS ##################

# Crea un'istanza della classe Grid
env = Grid()

episodes = 20

#La Q_table resta la stessa per diversi episodi
Q_table = np.zeros(shape=(25, 4), dtype=float)

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
    #print ("mapping_state input: ", next_state)
    new_pos = (next_state[1] * 5) + (next_state[0] * 1)

    return new_pos


for episode in range(episodes):
    print("######### NEW EPISODE ################")
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        # Ripristino il vettore delle azioni in modo da poterle eseguire tutte di nuovo
        env.action_space = np.array([0, 1, 2, 3])

        epsilon = 1 / env.path_length
        not_epislon = 1 - (1 / env.path_length)
        print ("stato iniziale: ", state)
        if random.uniform(0, 1) < epsilon:
            """ Explore: select a random action   """
            # Choose action
            action = random.choice(env.action_space)
            #next state ritornato dall'oracolo è già mappato
            do_action = env.oracle(state, action)
            yes = do_action
            state2 = mapping_state(state)
            #mapped_state = mapping_state(next_state)
            mapped_state = mapping_state(state)
            if yes: #L'azione resta nei boundaries, la eseguo
                print("Exploration action: ", action)
                # Perform action
                next_state, reward, done, info = env.step(action)

                # Aggiorna la Q_table
                #update_qtable(state2, action, reward, mapped_state)
                update_qtable(state2, action, reward, next_state)
                print("Stato: ", state2, " azione: ", action, " next state: ", next_state, " reward: ", reward)
                print ("q-table:\n", Q_table)
                score += reward
            else:  #ESCO DAI BOUNDARIES
                # Aggiorna la Q_table
                reward = -100
                #update_qtable(state2, action, reward, mapped_state)
                update_qtable(state2, action, reward, state2) # passo come next_state lo stato corrente -> NON è CORRETTO
                score += reward
                print ("Con l'azione ", action, " nello stato ", state2,
                       " vado fuori dai limiti del mondo")
                env.success = False
                done = True
            '''
            do_action = False
            while not do_action:
                action = random.choice(env.action_space)
                do_action = env.oracle(state, action)

            '''

        else:
            """ Exploit: select the action with max value (future reward)  """
            state2 = mapping_state(state)
            action_qvalues = Q_table[state2].tolist()
            max_action_value = max(action_qvalues)
            #Choose action
            action = action_qvalues.index(max_action_value)
            # next state ritornato dall'oracolo è già mappato
            do_action = env.oracle(state, action)
            #mapped_state = mapping_state(next_state)
            mapped_state = mapping_state(state)
            yes= do_action
            if yes: #L'azione resta nei boundaries, la eseguo
                print("Exploit action: ", action)
                # Perform action
                next_state, reward, done, info = env.step(action)
                # Aggiorna la Q_table
                #update_qtable(state2, action, reward, mapped_state)
                update_qtable(state2, action, reward, state2)
                print("Stato: ", state2, " azione: ", action, " next state: ", next_state, " reward: ", reward)
                print("q-table:\n", Q_table)
                score += reward
            else: #ESCO DAI BOUNDARIES
                reward = -100
                # Aggiorna la Q_table
                #update_qtable(state2, action, reward, mapped_state)
                update_qtable(state2, action, reward, state2) # Invece di next_state gli passo lo stato attuale -> NON CORRETTO
                score += reward
                print ("Con l'azione ", action, " nello stato ", state2,
                       " vado fuori dai limiti del mondo")
                env.success = False
                done = True
            '''
            print ("Exploit action: ", action)
            #Perform action
            next_state, reward, done, info = env.step(action)
            mapped_state = mapping_state(next_state)
            # Aggiorna la Q_table
            update_qtable(state2, action, reward, mapped_state)
            score += reward
            '''

    print('EPISODE:{} SCORE:{} STEP:{}, SUCCESS:{} \nFINAL Q_table:{}'.format(episode, score, env.path_length, env.success, Q_table))

print("Fine programma")
