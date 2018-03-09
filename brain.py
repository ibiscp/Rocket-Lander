from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import tensorflow as tf

class Brain:

    # Initialize brain
    def __init__(self, stateCnt, actionCnt, batchSize, learningRate):
        self.stateCnt = stateCnt                 # number of states
        self.actionCnt = actionCnt               # number os actions
        self.batchSize = batchSize               # batch size
        self.learningRate = learningRate

        self.model = self.createModel()          # model
        self.targetModel = self.createModel()    # target model

    # Huber loss function
    def huber_loss(self, y_true, y_pred, huber_loss_delta=1.0):

        err = y_true - y_pred

        cond = K.abs(err) < huber_loss_delta
        L2 = 0.5 * K.square(err)
        L1 = huber_loss_delta * (K.abs(err) - 0.5 * huber_loss_delta)

        loss = tf.where(cond, L2, L1)

        return K.mean(loss)

    # Create model
    def createModel(self):
        model = Sequential()
        model.add(Dense(units=64, activation='relu', input_dim=self.stateCnt))
        model.add(Dense(units=64, activation='relu'))
        # model.add(Dense(units=512, activation='relu'))
        # model.add(Dense(units=512, activation='relu'))
        model.add(Dense(units=self.actionCnt, activation='linear'))
        model.compile(loss=self.huber_loss, optimizer=RMSprop(lr=self.learningRate))
        return model

    # Train model using batch of random examples
    def train(self, x, y, batchSize, epochs=1, verbose=0):
        self.model.fit(x, y, batchSize, epochs=epochs, verbose=verbose)

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
