from numpy import zeros
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
#from keras.layers import Conv2D
#from keras.layers import Conv2DTranspose
#from keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras import initializers
from tensorflow.keras import Input
from tensorflow.keras import Model

import tensorflow as tf
import os
import numpy as np

from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)

tf.compat.v1.enable_eager_execution()
tf.config.optimizer.set_jit(True)

class DQN:
    def __init__(self, num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.Adam(lr=0.0001, amsgrad=True)
        self.gamma = gamma
        self.model = self.create_model((1, num_states))
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def create_model(self, input_shape):
        model = Sequential()
        model.add(Dense(8, input_shape=input_shape, kernel_initializer='RandomNormal'))
        model.add(Dense(16, kernel_initializer='RandomNormal'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(32, kernel_initializer='RandomNormal'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(16, kernel_initializer='RandomNormal'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(9, kernel_initializer='RandomNormal', activation='linear'))

        # opt = Adam(lr=0.0001, amsgrad=True)
        model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])
        return model

    def train(self, TargetNet):
        tf.compat.v1.enable_eager_execution()
        tf.config.optimizer.set_jit(True)
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(
            self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])

        # states = np.take(self.experience['s'], ids, axis=0)
        # actions = np.take(self.experience['a'], ids)
        # rewards = np.take(self.experience['r'], ids)
        # states_next = np.take(self.experience['s2'], ids, axis=0)
        # dones = np.take(self.experience['done'], ids)

        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                input_tensor=self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            loss = tf.math.reduce_mean(
                input_tensor=tf.square(actual_values - selected_action_values))
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())

    def save_model(self, path, epoch):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(path + f'/model_{epoch}.h5')

    def load_model(self, path):
        self.model.load_weights(path)
