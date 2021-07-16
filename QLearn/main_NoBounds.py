from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import time


import matplotlib

max_grid_length = 4
starting_grid_pos = 0

# global success_number
global success_number
success_number = 0


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
        # print ("oracolo in esecuzione!")

        if action == 0 and state[1] < max_grid_length:  # GO UP
            # actual_state[1] += 1
            return True

        elif action == 1 and state[1] > starting_grid_pos:  # GO DOWN
            # actual_state[1] -= 1
            return True

        elif action == 2 and state[0] > starting_grid_pos:  # GO LEFT
            # actual_state[0] -= 1
            return True

        elif action == 3 and state[0] < max_grid_length:  # GO RIGHT
            # actual_state[0] += 1
            return True

        else:
            #print("Non valid action!")
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
            reward = +2000
            # print("TARGET STATE RAGGIUNTO, termine episodio")

        elif new_target_distance < old_target_distance:
            reward = +1
        else:
            reward = -1

        # Check if search is done
        if self.path_length > episode_length:
            done = True
            # print("Fine episodio causa step terminati")
        elif self.state[0] == self.target_state[0] and self.state[1] == self.target_state[1]:
            # Se arrivo nello stato target allora l'episodio termina
            done = True
            self.success = True
            #print("Fine episodio causa target state raggiunto")
        else:
            done = False

        # Set placeholder for info
        info = {}
        target_distance = new_target_distance - old_target_distance
        #print("self.state: ", self.state, " distance from the target state: ", target_distance)
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


# Aggiornamento dei Q-values nella matrice Q_table
def update_qtable(state, action, reward, new_state, learning_rate, gamma):
    # Cumulative reward
    Q_table[state, action] += learning_rate * (reward + gamma * np.max(Q_table[new_state]) - Q_table[state, action])

    # print("Q_TABLE INPUTS: \n state: ", state, " action: ", action, " reward: ", reward, " new state: ", new_state,
    #       " learning rate: ", learning_rate, " gamma: ", gamma)

    row_numb = 0
    # print("#### Q TABLE #### \n")
    # for x in Q_table:
    #     print("row ", row_numb, " ", x)
    #     row_numb += 1

    return Q_table


# Mappa gli stati della matrice dell'enviorment in un vettore lungo 25, come i tutti i possibili stati
def mapping_state(next_state):
    # [0,0] -> 0; [0,1] -> 1 ... [1,0] -> 5
    # print ("mapping_state input: ", next_state)
    new_pos = (next_state[1] * 5) + (next_state[0] * 1)

    return new_pos


def run(learning_rate, gamma, episodes, episodes_length, run_number, Q_table):
    global state, success_number

    success_rate = 0
    output = []

    for episode in range(episodes):
        #print("######### NEW EPISODE ################")
        #print("Q_table new episode: \n", Q_table)
        state = env.reset()
        done = False
        score = 0

        # Action available
        env.action_space = np.array([0, 1, 2, 3])

        start = time.time()
        while not done:
            env.render()

            epsilon = 1 / env.path_length
            not_epislon = 1 - (1 / env.path_length)
            #print("stato iniziale: ", state)
            if random.uniform(0, 1) < epsilon:
                """ Explore: select a random action   """
                # Choose action
                action = random.choice(env.action_space)
                # print("Exploration action: ", action)
                # next state ritornato dall'oracolo è già mappato
                do_action = env.oracle(state, action)
                yes = do_action
                state2 = mapping_state(state)
                # mapped_state = mapping_state(next_state)
                mapped_state = mapping_state(state)
                if yes:  # L'azione resta nei boundaries, la eseguo

                    # Perform action
                    next_state, reward, done, info = env.step(action)

                    # Aggiorna la Q_table
                    # update_qtable(state2, action, reward, mapped_state)
                    update_qtable(state2, action, reward, next_state, learning_rate, gamma)
                    #print("Stato: ", state2, " azione: ", action, " next state: ", next_state, " reward: ", reward)
                    # print("q-table:\n", Q_table)
                    if reward == 2000:
                        success_number += 1

                    score += reward
                else:  # ESCO DAI BOUNDARIES
                    # Aggiorna la Q_table
                    reward = -100
                    # update_qtable(state2, action, reward, mapped_state)
                    update_qtable(state2, action, reward,
                                  state2, learning_rate,
                                  gamma)  # passo come next_state lo stato corrente -> NON è CORRETTO
                    score += reward
                    # print("Con l'azione ", action, " nello stato ", state2,
                    #       " vado fuori dai limiti del mondo")
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
                # Choose action
                actions = [i for i, x in enumerate(action_qvalues) if x == max_action_value]
                #print("Actions available: ", actions)
                action = random.choice(actions)
                #print("Exploit action: ", action)
                # action = action_qvalues.index(max_action_value)
                # next state ritornato dall'oracolo è già mappato
                do_action = env.oracle(state, action)
                # mapped_state = mapping_state(next_state)
                mapped_state = mapping_state(state)
                yes = do_action
                if yes:  # L'azione resta nei boundaries, la eseguo

                    # Perform action
                    next_state, reward, done, info = env.step(action)
                    # Aggiorna la Q_table
                    # update_qtable(state2, action, reward, mapped_state)
                    update_qtable(state2, action, reward, next_state, learning_rate, gamma)
                    #print("Stato: ", state2, " azione: ", action, " next state: ", next_state, " reward: ", reward)
                    # print("q-table:\n", Q_table)
                    if reward == 2000:
                        success_number += 1
                    score += reward
                else:  # ESCO DAI BOUNDARIES
                    reward = -100
                    # Aggiorna la Q_table
                    # update_qtable(state2, action, reward, mapped_state)
                    update_qtable(state2, action, reward,
                                  state2, learning_rate,
                                  gamma)  # Invece di next_state gli passo lo stato attuale -> NON CORRETTO
                    score += reward
                    #print("Con l'azione ", action, " nello stato ", state2,
                    #      " vado fuori dai limiti del mondo")
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
        # print("Episode score: ", score)
        end = time.time()
        # print('Run:{}, Episode:{}, Score:{} Steps:{}, Success:{}, ExecutionTime:{}, LearningRate:{}, Gamma:{}'.format(run_number,
        #                                                                                                     episode, score,
        #                                                                                                     env.path_length,
        #                                                                                                     env.success,
        #                                                                                                     end - start,
        #                                                                                                     learning_rate,
        #                                                                                                     gamma))

        success_rate = success_number / episodes
        # print ("FINAL Q TABLE OF THE EPSISODE: \n", Q_table)
        # print("Fine programma, success_rate: ", success_rate * 100, "%", "su ", episodes, " episodi",
        #        "; hanno avuto successo ", success_number, " episodi.")
        # print("End episode")

        output = [run_number, episode, score, env.path_length, env.success, end - start, learning_rate, gamma,
                  success_rate, Q_table]
        # for element in output:
        #     f.write(str(element))
        #     f.write(",")
        #
        # f.write('\n')
    success_rate = success_number / episodes

    #print("End run")
    return Q_table

############## TESTING ###################
# Nella fase di test, l'agente eseguirà sempre mosse di exploitation. La Q_table non deve essere aggiornata
# Mi interessano solo i risultati.
def test(episodes, episode_length, Q_table):
    # print ("##########################TESTING PHASE ##############################################\n")

    global state

    success_number = 0
    success_rate = 0
    episode_totalScore = 0
    episode_totalLength = 0

    output = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        score = 0

        # Action available
        env.action_space = np.array([0, 1, 2, 3])
        while not done:
            env.render()
            """ Exploit: select the action with max value (future reward)  """
            state2 = mapping_state(state)
            action_qvalues = Q_table[state2].tolist()
            max_action_value = max(action_qvalues)
            # Choose action
            actions = [i for i, x in enumerate(action_qvalues) if x == max_action_value]
            action = random.choice(actions)
            do_action = env.oracle(state, action)
            mapped_state = mapping_state(state)
            yes = do_action
            if yes:  # L'azione resta nei boundaries, la eseguo

                # Perform action
                next_state, reward, done, info = env.step(action)
                # Aggiorna la Q_table
                # update_qtable(state2, action, reward, next_state, learning_rate, gamma)
                if reward == 2000:
                    success_number += 1
                score += reward
            else:  # ESCO DAI BOUNDARIES
                reward = -100
                # Aggiorna la Q_table
                # update_qtable(state2, action, reward,
                #               state2, learning_rate,
                #               gamma)
                score += reward
                env.success = False
                done = True

        episode_totalScore += score
        episode_totalLength += env.path_length
        output = [episode, score, env.path_length, env.success, success_rate]
        # print ("EPISODE OUTPUT: Episode: {} , Score: {} , env.pathLength: {}  , env.success: {} , success_rate: {} ".format(output[0],
        #                                 output[1], output[2], output[3], output[4]))

    success_rate = success_number / episodes
    episode_AvgScore = episode_totalScore / episodes
    episode_avgLength = episode_totalLength /episodes

    episode_output= [success_rate, episode_AvgScore, episode_avgLength, env.success]
    return episode_output


if __name__ == "__main__":
    episode_length = 200
    learn_episodes = 70

    # Eseguendo azioni di tipo Exploit, il cammino dell'agente è deterministico, quindi non ha senso fare
    # Più episodi
    test_episode = 1

    # Learning rate: how much you accept the new value vs the old value.
    # Gamma: discount factor

    learning_rate = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 3.0, 4.0, 5.0]
    gamma = [0.1, 0.3, 0.5, 0.7, 0.9]

    with open('NewOutput.csv', 'w') as f:
        f.write("Test Number,Success,Score,Steps,Learning Rate,Gamma\n")
        best_score = 0
        best_QTABLE = np.zeros(shape=(25, 4), dtype=float)
        best_lr = 0
        best_gamma = 0
        best_length = episode_length + 2


        run_number = 0
        # Env inherited from Open AI Gym
        # Ciclo for per far girare i vari esperimenti
        for x in learning_rate:
            for y in gamma:
                # print("\n\n #######NUOVA RUN###############")
                # f.write('\n\nNUOVO CICLO\n')
                # La Q_table resta la stessa per diversi episodi
                Q_table = np.zeros(shape=(25, 4), dtype=float)

                results = []
                Q_table = run(x, y, learn_episodes, episode_length, run_number, Q_table)
                run_number += 1

                # episode_output = [success_rate, episode_AvgScore, episode_avgLength, env.success]
                results = test (test_episode, episode_length, Q_table)
                # Results[1] = avg_score
                actual_score = results[1]
                success = results[3]
                actual_length = results[2]

                # episode_output= [success_rate, episode_AvgScore, episode_avgLength]
                print ("Test Number: {}, success: {}, Avg_score: {}, AvgLength: {}, Learning Rate: {}, gamma: {}".format(
                                                                                                    run_number,
                                                                                                    results[0],
                                                                                                    results[1],
                                                                                                    results[2],
                                                                                                    x, y))

                final_results = []
                final_results.extend([run_number, results[0], results[1], results[2], x , y])

                for result in final_results:
                    f.write(str(result))
                    f.write(",")

                f.write("\n")


                # STRATEGIA 1 = Prendiamo La Q_table con il miglior AVG_Score
                # STRATEGIA 2 = Prendiamo La Q_table con il minor path_length AND success = True
                # Scegli strategia 1 o 2

                # STRATEGIA 2 = Prendiamo La Q_table con il minor path_length AND success = True
                if (success == True):
                    if (actual_length < best_length):
                        best_length = actual_length
                        best_score = actual_score
                        best_QTABLE = Q_table
                        best_gamma = y
                        best_lr = x
                    else:
                        pass

                    # STRATEGIA 1 = Prendiamo La Q_table con il miglior AVG_Score
                elif (best_score > actual_score):
                    pass
                else:
                    print ("Strategia 1\n")
                    best_score = actual_score
                    best_QTABLE = Q_table
                    best_gamma = y
                    best_lr = x



        print ("######### FINE TESTING ################### ")
        # Proviamo la Q_table migliore su griglie di dimensioni diverse e confrontiamo i risultati
        print ("\n BEST Q TABLE: \n", best_QTABLE, " \nhas the best score: ", best_score, " with learning rate: ", best_lr,
               " and gamma: ", best_gamma)





    f.close()
