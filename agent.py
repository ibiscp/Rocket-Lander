import random
import numpy as np
import math
import pickle
import glob
from memory import Memory
from brain import Brain

class Agent:

    # Initialize agent
    def __init__(self, stateCnt, actionCnt, memoryCapacity, updateTarget,
    batchSize, epsilonStart, epsilonEnd, epsilonDecay, learningRate, environment, gamma):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.batchSize = batchSize
        self.epsilon = epsilonStart
        self.epsilonStart = epsilonStart
        self.epsilonEnd = epsilonEnd
        self.epsilonDecay = epsilonDecay
        self.learningRate = learningRate
        self.memoryCapacity = memoryCapacity
        self.updateTarget = updateTarget
        self.environment = environment
        self.gamma = gamma
        self.uid = random.randint(0,100000)                 # identification
        self.episode = 1

        self.brain = Brain(self.stateCnt, self.actionCnt, self.batchSize, self.learningRate)  # initialize brain
        self.memory = Memory(self.memoryCapacity)                # initialize memory
        self.rewards = []                                   # list of rewards
        self.steps = 0                                      # total number of steps

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
    def appendReward(self, ep, steps, reward):
        self.rewards.append([ep, steps, reward])
        if ep != self.episode:
            self.episode = ep

    # Decrement epsilon
    def decrementEpsilon(self):
        self.steps += 1
        self.epsilon = self.epsilonEnd + (self.epsilonStart - self.epsilonEnd) * math.exp(-self.epsilonDecay * self.steps)

    # Replay saved data
    def replay(self):
        batch = self.memory.sample(self.batchSize)
        batchLen = len(batch)

        no_state = np.zeros(self.stateCnt)
        states = np.array([ o[0] for o in batch ])
        states_ = np.array([ (no_state if o[3] is None else o[3]) for o in batch ])

        p = self.brain.predict(states, target=False)
        p_ = self.brain.predict(states_, target=False)
        pTarget_ = self.brain.predict(states_, target=True)

        x = np.zeros((batchLen, self.stateCnt))
        y = np.zeros((batchLen, self.actionCnt))

        for i in range(batchLen):
            o = batch[i]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]

            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + self.gamma * pTarget_[i][ np.argmax(p_[i]) ]    # Double DQN
                #t[a] = r + self.gamma * np.amax(pTarget_[i])               # Target DQN
                #t[a] = r + self.gamma * np.amax(p_[i])                     # DQN

            x[i] = s
            y[i] = t

        self.brain.train(x, y, self.batchSize)

    # Save target model
    def saveModel(self, modelDir):
        file = modelDir + "/" + self.environment + "_" + str(self.episode)

        # Save model
        self.brain.saveModel( file + ".h5")

        # Save variables
        with open(file + '.pkl', 'wb') as f:
            pickle.dump([self.episode, self.epsilon, self.steps, self.uid, self.rewards, self.memory.samples], f)

    # Load model
    def loadModel(self, modelDir):
        try:
            # Get files
            files = glob.glob(modelDir + "/" + self.environment + "*.h5")
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
                self.episode, self.epsilon, self.steps, self.uid, self.rewards, self.memory.samples = pickle.load(f)

            self.episode += 1
            print("\n\nFile " + file.replace(modelDir + "/", "") + " succesfuly loaded")
            print("Continue training in episode " + str(self.episode) + "\n")

        except:
            print("\n\nNo file for the environment " + self.environment + " saved")
            print("Starting training in episode 1" + "\n")
