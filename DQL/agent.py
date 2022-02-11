import random
import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import adam_v2
from collections import deque
from DQL.utils import *


class MineSweeperAgent:
    def __init__(self, name, board_size, batch_size=8,
                 lr=0.01, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.9999, gamma=0.95,
                 conv_units=64, dense_units=256,
                 mem_size=5000):

        # Parameters
        self.name = name
        self.board_size = board_size
        self.actions_size = self.board_size * self.board_size
        self.mem_size = mem_size
        self.batch_size = batch_size

        self.lr = lr
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.conv_units = conv_units
        self.dense_units = dense_units

        # Experience replay
        self.memory = deque(maxlen=self.mem_size)

        # Init networks
        self.main_network = MineSweeperAgent.create_model(self.lr, [self.board_size, self.board_size, 1],
                                                          self.board_size * self.board_size,
                                                          self.conv_units, self.dense_units)

        self.target_network = MineSweeperAgent.create_model(self.lr, [self.board_size, self.board_size, 1],
                                                            self.board_size * self.board_size,
                                                            self.conv_units, self.dense_units)

    @staticmethod
    def create_model(lr, input_shape, output_dim, conv_units, dense_units):
        model = Sequential()
        model.add(Conv2D(filters=conv_units, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(filters=conv_units, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(filters=conv_units, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(filters=conv_units, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                         input_shape=input_shape))

        model.add(Flatten())
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dense(output_dim, activation='linear'))

        optimizer = adam_v2.Adam(learning_rate=lr, epsilon=1e-4)
        model.compile(optimizer=optimizer, loss='mse')

        return model

    def act(self, state):
        flatted_state = state.reshape(-1)
        unsolved = [i for i, x in enumerate(flatted_state) if x == -1]

        rand = np.random.rand()
        if rand < self.epsilon:
            move = np.random.choice(unsolved)
        else:
            moves = self.main_network.predict(np.expand_dims(state, 0))[0]
            moves[flatted_state != -1] = np.min(moves)

            move = np.argmax(moves)

        return idx_to_x_y(move, self.board_size)

    def replay(self):
        batch = random.sample(self.memory, self.batch_size)

        train_state = []
        train_target = []

        for state, action_idx, reward, next_state, done in batch:
            target = self.main_network.predict(np.expand_dims(state, 0))[0]

            if done:
                target[action_idx] = reward
            else:
                t = self.target_network.predict(np.expand_dims(state, 0))[0]
                target[action_idx] = reward + self.gamma * np.amax(t)

            train_state.append(state)
            train_target.append(target)

        self.main_network.fit(np.array(train_state), np.array(train_target), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, x_y_to_idx(action[0], action[1], self.board_size),
                            reward, next_state, done))

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def load(self, name):
        self.main_network.load_weights(name)
        self.update_target_network()

    def save(self, name):
        self.target_network.save_weights(name)
