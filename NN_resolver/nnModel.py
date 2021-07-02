import numpy
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# The network model is a dense type, so each layer is fully connected with the others
# The first two layers has ReLU activation function.
# The output layer has Linear function, that return an array with 4 probs of 4 actions.
def build_model(states, actions):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=states))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

# Build agent with policy
# return a trained agent with Q function
def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=5000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=1000,
                   target_model_update=1e-2)
    return dqn
