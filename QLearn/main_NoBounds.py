from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
import time


max_grid_length = 4
starting_grid_pos = 0

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
            #NON VALID ACTION
            return False


    def step(self, action):
        # action| path_length | state
        # 0     |  +1         | (_, +1)
        # 1     |  +1         | (_, -1)
        # 2     |  +1         | (-1, _)
        # 3     |  +1         | (+1, _)

        # Euclidean distance calculation of the "old state":
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
        new_target_distance = np.linalg.norm(self.state - self.target_state)

        # Calculate reward
        if self.state[0] == self.target_state[0] and self.state[1] == self.target_state[1]:
            reward = +2000
            # TARGET STATE REACHED, end episode

        elif new_target_distance < old_target_distance:
            # Getting closer to the target state
            reward = +1
        else:
            # Walking away from the target state
            reward = -1

        # Check if search is done
        if self.path_length > episode_length:
            done = True
            # Steps fineshed, end episode
        elif self.state[0] == self.target_state[0] and self.state[1] == self.target_state[1]:
            # Target state reached, end episode
            done = True
            self.success = True
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
        self.state = np.array([starting_grid_pos, starting_grid_pos])
        # Reset path
        self.path_length = 1
        return self.state


########## END GRID CLASS ##################


env = Grid()


# Updating the Q-values in the Q_table array
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


# Maps the states of the environment matrix into a vector of length 25, such as all possible states
def mapping_state(next_state):
    # [0,0] -> 0; [0,1] -> 1 ... [1,0] -> 5
    # print ("mapping_state input: ", next_state)
    new_pos = (next_state[1] * 5) + (next_state[0] * 1)

    return new_pos


############## LEARNING PHASE ############################
def run(learning_rate, gamma, episodes, episodes_length, run_number, Q_table):
    #print ("Learning Phase")
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

        while not done:
            env.render()

            epsilon = 1 / env.path_length
            not_epislon = 1 - (1 / env.path_length)
            if random.uniform(0, 1) < epsilon:
                """ Explore: select a random action   """
                # Choose action
                action = random.choice(env.action_space)
                # print("Exploration action: ", action)
                do_action = env.oracle(state, action)
                yes = do_action
                state2 = mapping_state(state)
                mapped_state = mapping_state(state)
                if yes:  # The action stays in the boundaries, I execute it

                    # Perform action
                    next_state, reward, done, info = env.step(action)

                    # Update the Q_table
                    update_qtable(state2, action, reward, next_state, learning_rate, gamma)
                    #print("state: ", state2, " action: ", action, " next state: ", next_state, " reward: ", reward)

                    # print("q-table:\n", Q_table)
                    if reward == 2000:
                        success_number += 1

                    score += reward
                else:  # Agent go out of boundiries
                    # Update the Q_table
                    reward = -100
                    update_qtable(state2, action, reward,
                                  state2, learning_rate,
                                  gamma)  # send the current state as next_state
                    score += reward
                    env.success = False
                    done = True

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
                do_action = env.oracle(state, action)
                mapped_state = mapping_state(state)
                yes = do_action
                if yes:  # The action stays in the boundaries, I execute it

                    # Perform action
                    next_state, reward, done, info = env.step(action)
                    # Update the Q_table
                    update_qtable(state2, action, reward, next_state, learning_rate, gamma)
                    #print("state: ", state2, " action: ", action, " next state: ", next_state, " reward: ", reward)
                    # print("q-table:\n", Q_table)
                    if reward == 2000:
                        success_number += 1
                    score += reward
                else:  # Agent go out of boundaries
                    reward = -100
                    # Update the Q_table
                    update_qtable(state2, action, reward,
                                  state2, learning_rate,
                                  gamma)  # send the current state as next_state
                    score += reward
                    env.success = False
                    done = True
    return Q_table

############## TESTING ###################
# In the testing phase, the agent will always perform exploitation moves. The Q_table does not need to be updated
def test(episodes, episode_length, Q_table):

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
    learn_episodes = 100

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


        for x in learning_rate:
            for y in gamma:
                Q_table = np.zeros(shape=(25, 4), dtype=float)

                results = []
                Q_table = run(x, y, learn_episodes, episode_length, run_number, Q_table)
                run_number += 1

                # results = [success_rate, episode_AvgScore, episode_avgLength, env.success]
                results = test (test_episode, episode_length, Q_table)

                actual_score = results[1]
                success = results[3]
                actual_length = results[2]

                print("Test Number: {}, success: {}, Avg_score: {}, AvgLength: {}, Learning Rate: {}, gamma: {}".format(
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


                # STRATEGY 1 = Let's take The Q_table with the best AVG_Score
                # STRATEGY  2 = Let's take The Q_table with the lowest path_length AND success = True
                if (success == True):
                    if (actual_length < best_length):
                        # STRATEGY 2
                        best_length = actual_length
                        best_score = actual_score
                        best_QTABLE = Q_table
                        best_gamma = y
                        best_lr = x
                    else:
                        pass


                elif (best_score > actual_score):
                    pass
                else:
                    # STRATEGY 1
                    best_score = actual_score
                    best_QTABLE = Q_table
                    best_gamma = y
                    best_lr = x



        print ("\nEnd Testing \n")
        print ("\n BEST Q TABLE: \n", best_QTABLE, " \nhas the best score: ", best_score, " with learning rate: ", best_lr,
               " and gamma: ", best_gamma)





    f.close()
