import numpy
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten


from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from tensorflow.python.keras.layers import Reshape


def build_model(states, actions):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(2,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


def build_agent(model,actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=10, target_model_update = 1e-2)
    return dqn