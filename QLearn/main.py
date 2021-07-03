from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random

import matplotlib

'''
 TODO: 
    Task 1:
    - Fix Seeker policy, movements are defined and his utilities too.
    - Seeker KNOWS Hider's position
    - Assume Hider knows where Seeker is
    - Learn strategy to maximize HIDING
    
    Task 2: 
    - Hider has fixed policy and Seeker has to find him
    - Hider has random policy and Seeker has to find him
    - Differences between the two results
    
    Extra Task:
    - Explorative operations on maze with one/two approaches:
        ° TOP-DOWN (DFS or BFS)
        ° CLP   
    - Possible graphical presentation with pygame and tutorial

'''

max_grid_length = 4
starting_grid_pos = 0
episode_length = 100

# Env inherited from Open AI Gym
class Grid(Env):

    def __init__(self):

        # Actions: up, down, left, right
        self.action_space = Discrete(4)
        # grid cells
        self.observation_space = Box(low=np.array([starting_grid_pos,starting_grid_pos]), high=np.array([max_grid_length, max_grid_length]), dtype=int)

        # start position
        self.state = np.array([starting_grid_pos,starting_grid_pos])

        # starting episode length
        self.path_length = 1

        self.wall = False

        self.target_state = np.array([max_grid_length, max_grid_length])


    def step(self, action):
        # action| path_length | state
        # 0     |  +1         | (_, +1)
        # 1     |  +1         | (_, -1)
        # 2     |  +1         | (-1, _)
        # 3     |  +1         | (+1, _)
        #self.state è nella forma [x,y]

        #Calcolo distanza euclidea del "vecchio stato":
        old_target_distance = np.linalg.norm(self.state - self.target_state)

        if action == 0: #GO UP
            self.state[1] += 1
            if (state[1] <= max_grid_length):
                # Il movimento resta all'interno della griglia
                pass
                #print (self.state)

            else: #Il movimento esce dai limiti della griglia:
                print ("mossa in alto non valida!")
                #Ritorno allo stato precedente
                self.state[1] -= 1
                #Tolgo l'ozione 'up' dai possibili movimenti
                #NON SONO SICURO!!
                env.action_space = np.array([1, 2, 3]) #Mi ha modificato il vettore delle possibili opzioni per tutte le altre mosse?
                #Ri-eseguo lo step senza la mossa illegale
                #action = env.action_space.sample()
                action = random.choice(env.action_space) #Ricalcolo un'azione random sul nuovo array di azioni disponibili
                next_state, reward, done, info = env.step(action)
                print("New action: ", action, " new state: ", next_state, " self.state: ", self.state)

        if action == 1: #GO DOWN

            self.state[1] -= 1
            if (state[1] >= starting_grid_pos):
                # Il movimento resta all'interno della griglia
                pass
                #print(self.state)
            else:
                # Il movimento esce dai limiti della griglia:
                print ("Mossa in basso non valida!")
                self.state[1] += 1 #Ritorno allo stato precedente
                env.action_space = np.array([0, 2, 3])
                action = random.choice(env.action_space)
                next_state, reward, done, info = env.step(action)
                print("New action: ", action, " new state: ", next_state, " self.state: ", self.state)

        if action == 2: #GO LEFT
            self.state[0] -= 1
            if (state[0] >= starting_grid_pos):
                # Il movimento resta all'interno della griglia
                pass
                #print(self.state)
            else:
                print("Mossa a sinistra non valida!")
                self.state[0] += 1
                env.action_space = np.array([0, 1, 3])
                action = random.choice(env.action_space)
                next_state, reward, done, info = env.step(action)
                print("New action: ", action, " new state: ", next_state , " self.state: ", self.state)

        if action == 3: #GO RIGHT
            self.state[0] += 1
            if (state[0] <= max_grid_length):
                # Il movimento resta all'interno della griglia
                pass
                #print(self.state)
            else:
                print("Mossa a destra non valida!")
                self.state[0] -= 1
                env.action_space = np.array([0, 1, 2])
                action = random.choice(env.action_space)
                next_state, reward, done, info = env.step(action)
                #print("New action: ", action, " new state: ", next_state, " self.state: ", self.state)

        # Grow path length by 1
        self.path_length += 1


        # Calculate the euclidian distance from the current stat to the target state
        #Calcolo la distanza euclidea del nuovo stato
        new_target_distance = np.linalg.norm(self.state - self.target_state)
        print("new_target_distance: ", new_target_distance, " old_target_distance: ", old_target_distance)

        # Calculate reward
        if self.state[0] == self.target_state[0] and self.state[1] == self.target_state[1]:
            reward = 100
            print("TARGET STATE RAGGIUNTO, termine episodio")
        elif new_target_distance <= old_target_distance:
            reward = -1
        else:
            reward = 1

         # Check if search is done
        if self.path_length > episode_length:
            done = True
        elif self.state[0] == self.target_state[0] and self.state[1] == self.target_state[1]:
            #Se arrivo nello stato target allora l'episodio termina
            done= True
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

#Crea un'istanza della classe Grid
env = Grid()
print("OBSERVATION SPACE: ", env.observation_space.shape)
print("OBSERVATION SAMPLE: ", env.observation_space.sample())

episodes = 10

# Initialize q-table values to 0
# Q = np.zeros((state_size, action_size)). state_size = matrice "appiattita"
# ho 25 possibili stati e 4 possibili azioni
#Q_table = np.zeros(shape=(26, 4), dtype=float)
#print ("Initial Q_table: ", Q_table, print( "lunghezza asse x q table: ", len(Q_table[0])))

#Learning rate: how much you accept the new value vs the old value.
lr = 0.8

#Gamma: discount factor
gamma = 0.9

#Rewards: is the value received after completing a certain action at a given state.
#la reward è tornata dalla funzione env.step(action)

#Max: np.max() uses the numpy library and is taking the maximum of the future reward
# and applying it to the reward for the current state. What this does is impact the current
# action by the possible future reward. This is the beauty of q-learning.
# We’re allocating future reward to current actions to help the agent select the highest return action at any given state


#Aggiornamento dei Q-values nella matrice Q_table
def update_qtable(state, action, reward, new_state):

    #Cumulative reward
    #DA RIVEDERE VALORE DI LR (aka learning rate)
    #Q_table[state,action] = Q_table[state, action] + (lr * (reward + gamma * np.max(Q_table[new_state]) - Q_table[state,action]))
    Q_table[state,action] += lr * (reward + gamma * np.max(Q_table[new_state]) - Q_table[state,action])

    print("valore q_value aggiornato: ", Q_table[state, action], " nello stato: ", state, " con l'azione: ", action,
           "new state: ", new_state)
    return Q_table

#Mappa gli stati della matrice dell'enviorment in un vettore lungo 25, come i tutti i possibili stati
def mapping_state (next_state):
    #[0,0] -> 0; [0,1] -> 1 ... [1,0] -> 5
    #[3,3] -> 18
    # Se cambio la riga allora faccio +/- 5, se cambio la colonna +/- 1
    #print("mapping_state:  next_State[0]: ", next_state[0], " next_state[1]: ", next_state[1])

    new_pos= (next_state[1] * 5) + (next_state[0] * 1)
    #vect = np.arange(24) #array di 25 elementi

    #Per tornare alla matrice 5x5
    #matrix= vect.reshape(5,5)
    return new_pos

for episode in range(episodes):
    state = env.reset()
    done = False
    score = 0
    #La q_table rimane la stessa con episodi diversi FAKE NEWS
    #qtable_dimension = max_grid_length * max_grid_length

    # Resett Q_table ad ogni episodio
    Q_table = np.zeros(shape=(25, 4), dtype=float)

    print ("######### NEW EPISODE ################")

    while not done:
        env.render()
        #Ripristino il vettore delle azioni in modo da poterle eseguire tutte di nuovo
        env.action_space = np.array([0, 1, 2, 3])
        #Stato iniziale = state (self.state)
        #action = policy(state)
        #states = 0
        #action = None
        # Set the percent you want to explore | epsilon =0.2

        epsilon = 1/env.path_length
        not_epislon = 1 - (1/env.path_length)
        print('#########STEP:{}'.format(env.path_length))
        print("Probabilità di exploration: ", epsilon, " Probabilità di exploitation: ", not_epislon)
        print("Q TABLE: \n", Q_table)

        if random.uniform(0, 1) < epsilon:
            """ Explore: select a random action   """
            # Instead of selecting actions based on the max future reward we select an action at random.
            #action = env.action_space.sample()
            action = random.choice(env.action_space)
            state2 = mapping_state(state)
            print("action by exploring: ", action)
            next_state, reward, done, info = env.step(action)
            mapped_state = mapping_state(next_state)
            print("mapped actual state: ", state2, "cartesian actual state: ", state,
                " action/index to do: ", action)
            # print ( "new state: ", next_state, "mapped new state: ", mapping_state(next_state))
            # Aggiorna la Q_table
            new_Qtable = update_qtable(state2, action, reward, mapped_state)
            #print("new q table: ", new_Qtable)
            score += reward
        else:
            """ Exploit: select the action with max value (future reward)  """
            # The first is to use the q-table as a reference and view all possible actions for a given state.
            # The agent then selects the action based on the max value of those actions.

            # Per ogni stato ho 4 possibili opzioni. Infatti la Q_table è costruita apposta
            # Ognuno dei 25 stati ha 4 azioni disponibili con diversi pesi

            #CODICE:
            #Prendi l'azione migliore possibile in quello stato guardando la Q_table
            #max_action_value = int(np.max(Q_table[state]))
            state2 = mapping_state(state)
            action_qvalues = Q_table[state2].tolist()
            max_action_value = max(action_qvalues)
            action = action_qvalues.index(max_action_value)
            print("action by exploit: ", action)
            print("mapped actual state: ",mapping_state(state), "cartesian actual state: ", state,
                  "action_qvalues: ", action_qvalues, "max_action_value: ", max_action_value,
                  " action/index to do: ", action)

            next_state, reward, done, info = env.step(action)
            mapped_state = mapping_state (next_state)
            #print ("new_state: ", mapped_state, "mapped state: ", next_state, " reward: ", reward, " done: ", done)
            #Aggiorna la Q_table
            new_Qtable = update_qtable(state2,action, reward,  mapped_state)
            #print("new q table: ", new_Qtable)

        # action = env.action_space.sample()
        # next_state, reward, done, info = env.step(action)
        # score += reward
    print('#########EPISODE:{} SCORE:{}#############'.format(episode, score))
    print ("\n##################FINAL Q_table: ##################\n", Q_table)


print ("Fine programma")


