import random
import gym
import sys
import numpy as np
from collections import deque,namedtuple
import os
import time
import matplotlib.pyplot as plt
from keras import models
from keras import layers
from tensorflow.keras.optimizers import Adam

class AgentMountainCar:
    def __init__(self,env):
        self.env=env
        self.gamma=None

        self.epsilon = 1
        self.epsilon_decay = None
        self.epsilon_min=0.01

        self.learningRate=0.001
        self.replayBuffer=deque(maxlen=10000)

        self.episodeNum=None
        self.iterationNum=200
        self.BufferSize=None
        self.training_time = None

        self.rewards = []

    def get_train_time(self):
        return self.training_time

    def init_hyperparameters(self, n_episodes, batch_size, gamma, lr, decay):
        self.episodeNum = n_episodes
        self.BufferSize = batch_size
        self.gamma = gamma
        self.learningRate = lr
        self.epsilon_decay = decay
        self.trainNetwork=self.createNetwork()
        self.targetNetwork=self.createNetwork()
        self.targetNetwork.set_weights(self.trainNetwork.get_weights())

    def createNetwork(self):
        model = models.Sequential()
        state_shape = self.env.observation_space.shape

        model.add(layers.Dense(24, activation='relu', input_shape=state_shape))
        model.add(layers.Dense(48, activation='relu'))
        model.add(layers.Dense(self.env.action_space.n,activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learningRate))
        return model

    def getBestAction(self,state):

        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            action=np.argmax(self.trainNetwork.predict(state)[0])

        return action



    def trainFromBuffer(self):
        if len(self.replayBuffer) < self.BufferSize:
            return

        samples = random.sample(self.replayBuffer,self.BufferSize)

        states = []
        newStates=[]
        for sample in samples:
            state, action, reward, new_state, done = sample
            states.append(state)
            newStates.append(new_state)

        newArray = np.array(states)
        states = newArray.reshape(self.BufferSize, 2)

        newArray2 = np.array(newStates)
        newStates = newArray2.reshape(self.BufferSize, 2)

        targets = self.trainNetwork.predict(states)
        new_state_targets=self.targetNetwork.predict(newStates)

        i=0
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = targets[i]
            if done:
                target[action] = reward
            else:
                Q_future = max(new_state_targets[i])
                target[action] = reward + Q_future * self.gamma
            i+=1

        self.trainNetwork.fit(states, targets, epochs=1, verbose=0)


    def trainEpisode(self,currentState,eps):
        rewardSum = 0
        max_position=-99

        for i in range(self.iterationNum):
            bestAction = self.getBestAction(currentState)

            #show the animation every 10 eps
            if eps%10==0:
                self.env.render()

            new_state, reward, done, _ = self.env.step(bestAction)

            new_state = new_state.reshape(1, 2)

            # # Keep track of max position
            if new_state[0][0] > max_position:
                max_position = new_state[0][0]

            # # Adjust reward for task completion
            if new_state[0][0] >= 0.5:
                reward += 10000

            # Add reward for swinging
            elif (new_state[0][0] > currentState[0][0]) and (new_state[0][0] > -0.5) and (currentState[0][0] > -0.5):
                reward += 20

            elif (new_state[0][0] < currentState[0][0]) and (new_state[0][0] < -0.6) and (currentState[0][0] < -0.6):
                reward += 20

            # More penalty to not do any of above
            else:
                reward -= 10

            self.replayBuffer.append([currentState, bestAction, reward, new_state, done])

            #Or you can use self.trainFromBuffer_Boost(), it is a matrix wise version for boosting
            self.trainFromBuffer()

            rewardSum += reward

            currentState = new_state

            if done:
                break

        if i < 199:
            print("Success!!!! in episode {}. Current reward: {}. Current Max Position: {}!".format(eps, rewardSum, max_position))
            self.trainNetwork.save('./trainNetworkInEPS{}.h5'.format(eps))
        elif (i==199) and (eps%10 ==0) and (eps > 0):
            print("Failure in episode {}. Current reward: {}. Current Max Position: {}!".format(eps, rewardSum, max_position))


        #Sync
        self.targetNetwork.set_weights(self.trainNetwork.get_weights())
        self.epsilon *= self.epsilon_decay
        self.rewards.append(rewardSum)

    def train(self):
        t1 = time.time()
        for eps in range(self.episodeNum):
            currentState=self.env.reset().reshape(1,2)
            self.trainEpisode(currentState, eps)
            if eps == (self.episodeNum - 1):
                self.training_time = time.time() - t1
