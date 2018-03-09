import numpy as np
import gym
from gym import wrappers as wr
import math
import glob
import pickle
import time
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import tensorflow as tf
from collections import deque
import random

# Agent
gamma = 0.99                        # Reward discount factor
learning_rate = 0.00025             # Learning rate
num_episodes = 10000                # number of episodes
max_steps_ep = 1000000              # default max number of steps per episode (unless env has a lower hardcoded limit)
update_target = 1000                # number of steps to use slow target as target before updating it to latest weights
epsilon_start = 1.0                 # probability of random action at start
epsilon_end = 0.01                  # minimum probability of random action after linear decay period
epsilon_decay = 0.000001             # speed of decay
save_model_episode = 100            # interval to save model

# Brain
huber_loss_delta = 1.0              # huber loss delta
batch_size = 128                    # size of batch from experience replay memory for updates

# Memory
memory_capacity = 100000            # capacity of experience replay memory

# Environment
#'RocketLander-v0'
environment = 'Breakout-ram-v0'     # Environment name

# folders
monitorDir = 'videos'
modelDir = 'models'

class Brain:

    # Initialize brain
    def __init__(self, stateCnt, actionCnt, batchSize):
        self.stateCnt = stateCnt                 # number of states
        self.actionCnt = actionCnt               # number os actions
        self.batchSize = batchSize               # batch size

        self.model = self.createModel()          # model
        self.targetModel = self.createModel()    # target model

    # Huber loss function
    def huber_loss(self, y_true, y_pred):

        err = y_true - y_pred

        cond = K.abs(err) < huber_loss_delta
        L2 = 0.5 * K.square(err)
        L1 = huber_loss_delta * (K.abs(err) - 0.5 * huber_loss_delta)

        loss = tf.where(cond, L2, L1)

        return K.mean(loss)

    # Create model
    def createModel(self):
        model = Sequential()
        model.add(Dense(units=512, activation='relu', input_dim=stateCnt))
        model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=512, activation='relu'))
        # model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=actionCnt, activation='linear'))
        model.compile(loss=self.huber_loss, optimizer=RMSprop(lr=learning_rate))
        return model

    # Train model using batch of random examples
    def train(self, x, y, batchSize=batch_size, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=batchSize, epochs=epochs, verbose=verbose)

    # Predict using normal or target model given a batch of states
    def predict(self, s, target=False):
        if target:
            return self.targetModel.predict(s)
        else:
            return self.model.predict(s)

    # Predict given only one state
    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.stateCnt), target=target).flatten()

    # Update target model
    def updateTargetModel(self):
        self.targetModel.set_weights(self.model.get_weights())

    # Save target model
    def saveModel(self, file):
        self.targetModel.save(file)

    # Load weights
    def loadWeights(self, file):
        self.model.load_weights(file)
        self.targetModel.load_weights(file)

class Memory:

    # Initialize memory
    def __init__(self, capacity):
        self.samples = deque(maxlen=capacity)
        self.capacity = capacity

    # Add sample to the memory
    def add(self, sample):
        self.samples.append(sample)

    # Generate 'n' random samples from the memory
    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    # Number of current samples in memory
    def numberSamples(self):
        return len(self.samples)

class Agent:
    steps = 0
    epsilon = epsilon_start

    # Initialize agent
    def __init__(self, stateCnt, actionCnt, memoryCapacity, updateTarget, batchSize):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.updateTarget = updateTarget
        self.uid = random.randint(0,100000)                 # identification
        self.episode = 1

        self.brain = Brain(stateCnt, actionCnt, batchSize)  # initialize brain
        self.memory = Memory(memoryCapacity)                # initialize memory
        self.rewards = []                                   # list of rewards

    # Act based ond epsilon
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return np.argmax(self.brain.predictOne(s))

    # Save experience update target model if necessary
    def observe(self, sample):
        self.memory.add(sample)

        if (self.steps % self.updateTarget == 0):
            self.brain.updateTargetModel()

    # Append reward and update episode
    def appendReward(self, reward):
        self.rewards.append(reward)
        self.episode += 1

    # Decrement epsilon
    def decrementEpsilon(self, done):
        self.steps += 1
        self.epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-epsilon_decay * self.steps)

    # Replay saved data
    def replay(self):
        batch = self.memory.sample(batch_size)
        batchLen = len(batch)

        no_state = np.zeros(self.stateCnt)
        states = np.array([ o[0] for o in batch ])
        #states_ = np.array([ o[3] for o in batch ])
        states_ = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_, target=True)

        x = np.zeros((batchLen, self.stateCnt))
        y = np.zeros((batchLen, self.actionCnt))

        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]

            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + gamma * np.amax(p_[i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)

    # Save target model
    def saveModel(self):
        file = modelDir + "/" + environment + "_" + str(self.episode)

        # Save model
        self.brain.saveModel( file + ".h5")

        # Save variables
        with open(file + '.pkl', 'wb') as f:
            pickle.dump([self.episode, self.epsilon, self.steps, self.uid, self.rewards], f)

    # Load model
    def loadModel(self):
        try:
            # Get files
            files = glob.glob(modelDir + "/" + environment + "*.h5")
            episode = -1

            # Get last file and episode
            for f in files:
                ep = int(f.split("_",1)[1].replace(".h5", ""))
                if ep > episode:
                    file = f
                    episode = ep

            # Load NN weights
            self.brain.loadWeights(file)

            # Read variables
            with open( file.replace(".h5","") + '.pkl', 'rb') as f:
                self.episode, self.epsilon, self.steps, self.uid, self.reward = pickle.load(f)

            print("\n\nFile " + file.replace(modelDir + "/", "") + " succesfuly loaded")
            print("Continue training in episode " + str(episode) + "\n")

        except:
            print("\n\nNo file for the environment " + environment + " saved")
            print("Starting training in episode 1" + "\n")

# Start environment
env = gym.make(environment)

# State and action variables
stateCnt  = env.env.observation_space.shape[0]
actionCnt = env.env.action_space.n

# set seeds to 0
env.seed(0)
np.random.seed(0)

# Initialize agent
agent = Agent(stateCnt, actionCnt, memory_capacity, update_target, batch_size)

# Load model if exists
agent.loadModel()

# Populate memory
while agent.memory.numberSamples() <= batch_size:
    state = env.reset()
    done = False

    while not done:
        # act
        action = agent.act(state)

        # execute action
        next_state, reward, done, _ = env.step(action)

        # observe
        agent.memory.add((state, action, reward, None if done else next_state))
        #agent.memory.add((state, action, reward, np.zeros(self.stateCnt) if done else next_state))

        # update next sate
        state = next_state

# Reset environment episode
env = wr.Monitor(env, monitorDir, resume=True, video_callable=lambda episode_id: episode_id%100==0 or episode_id==1, uid=agent.uid)
env.episode_id = agent.episode

# Train the model
for ep in range(agent.episode, num_episodes + 1):
    total_reward = 0
    steps_in_episode = 0

    try:
        state = env.reset()
    except:
        time.sleep(1)
        state = env.reset()
        print("Erro mais foi!!")
    done = False

    while not done:
        # act
        action = agent.act(state)

        # execute action
        next_state, reward, done, _ = env.step(action)

        # update total reward
        total_reward += reward

        # observe
        agent.observe((state, action, reward, None if done else next_state))

        # decrement epsilon
        agent.decrementEpsilon(done)

        # replay
        agent.replay()

        # update variables
        state = next_state
        steps_in_episode += 1

        if done or steps_in_episode > max_steps_ep:
            done = True
            agent.appendReward(total_reward)

    # save model
    if (ep%save_model_episode==0):
        agent.saveModel()

    print('Episode %4i, Reward: %8.3f, Steps: %4i, Next eps: %6.4f'%(ep,total_reward,steps_in_episode, agent.epsilon))
