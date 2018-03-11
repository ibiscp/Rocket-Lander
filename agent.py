import random
import numpy as np
import math
import pickle
import glob
from memory import Memory
from brain import Brain

class Agent:
    steps = 0

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
        x, y, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)

        if (self.steps % self.updateTarget == 0):
            self.brain.updateTargetModel()

    # Append reward and update episode
    def appendReward(self, ep, reward):
        self.rewards.append([ep, reward])
        if ep != self.episode:
            self.episode = ep

    # Decrement epsilon
    def decrementEpsilon(self, done):
        self.steps += 1
        self.epsilon = self.epsilonEnd + (self.epsilonStart - self.epsilonEnd) * math.exp(-self.epsilonDecay * self.steps)

    def _getTargets(self, batch):
        batchLen = len(batch)

        no_state = np.zeros(self.stateCnt)

        states = np.array([ o[1][0] for o in batch ])
        #states_ = np.array([ o[1][3] for o in batch ])
        states_ = np.array([ (no_state if o[1][3] is None else o[1][3]) for o in batch ])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_, target=False)
        pTarget_ = self.brain.predict(states_, target=True)

        x = np.zeros((batchLen, self.stateCnt))
        y = np.zeros((batchLen, self.actionCnt))

        errors = np.zeros(batchLen)

        for i in range(batchLen):
            o = batch[i][1]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]

            t = p[i]
            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + self.gamma * pTarget_[i][ np.argmax(p_[i]) ]

            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])

        return (x, y, errors)

    # Replay saved data
    def replay(self):
        batch = self.memory.sample(self.batchSize)

        x, y, errors = self._getTargets(batch)

        #update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.brain.train(x, y, self.batchSize)

    # Save target model
    def saveModel(self, modelDir):
        file = modelDir + "/" + self.environment + "_" + str(self.episode)

        # Save model
        self.brain.saveModel( file + ".h5")

        # Save variables
        with open(file + '.pkl', 'wb') as f:
            pickle.dump([self.episode, self.epsilon, self.steps, self.uid, self.rewards, self.memory.tree], f)

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
                self.episode, self.epsilon, self.steps, self.uid, self.reward, self.memory.tree = pickle.load(f)

            self.memory.len = self.memory.tree.capacity
            self.episode += 1
            print("\n\nFile " + file.replace(modelDir + "/", "") + " succesfuly loaded")
            print("Continue training in episode " + str(episode) + "\n")

        except:
            print("\n\nNo file for the environment " + self.environment + " saved")
            print("Starting training in episode 1" + "\n")
