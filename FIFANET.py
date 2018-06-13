import numpy as np

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Dense, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import keras.backend as K

from callbacks import PlotLosses

import config as C

class FIFANET():
  def __init__(self):
    self.keyShape = C.INPUT_SHAPE
    self.input_shape = (self.keyShape,1)
    self.output_shape = (2,1)
    self.dropout_amount = C.DROPOUT
    self.optimizer = Adam(lr=C.LEARNING_RATE)

    self.model = self.buildNetwork()
    self.model.compile(loss=self.loss_function, optimizer=self.optimizer) #mean_squared_error

  def loss_function(self, y_real, y_pred):
    mse = K.mean(K.square(y_pred - y_real), axis=-1)

    correct_winner_regularization = 0 if K.sign(y_pred[0]-y_pred[1]) == K.sign(y_real[0]-y_real[1]) else C.WRONG_WINNER_PENALTY
    loss = mse + correct_winner_regularization

    return loss

  def buildNetwork(self):
    model = Sequential()

    model.add(Dense(512, input_dim=self.keyShape))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=C.LEAKY_RELU_ALPHA))
    #model.add(Dropout(self.dropout_amount))

    model.add(Dense(256, input_dim=self.keyShape))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=C.LEAKY_RELU_ALPHA))
    #model.add(Dropout(self.dropout_amount))

    model.add(Dense(256))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=C.LEAKY_RELU_ALPHA))
    model.add(Dropout(self.dropout_amount))

    model.add(Dense(128))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=C.LEAKY_RELU_ALPHA))
    model.add(Dropout(self.dropout_amount))

    model.add(Dense(128))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=C.LEAKY_RELU_ALPHA))
    model.add(Dropout(self.dropout_amount))

    model.add(Dense(64))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=C.LEAKY_RELU_ALPHA))
    #model.add(Dropout(self.dropout_amount))

    model.add(Dense(16))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=C.LEAKY_RELU_ALPHA))
    #model.add(Dropout(self.dropout_amount))

    model.add(Dense(2, activation="relu"))

    model.summary()

    return model

  def train(self, x_train, y_train, x_test, y_test, epochs=100, batch_size=128):

    cb_early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=100, verbose=1)
    #cb_plot_losses = PlotLosses()

    self.model.fit(x_train, y_train,
      epochs = epochs,
      batch_size = batch_size,
      validation_data = (x_test, y_test),
      shuffle = True,
      callbacks = [
        cb_early_stopping,
        #cb_plot_losses
      ]
    )
    score = self.model.evaluate(x_test, y_test, batch_size=batch_size)
    print('test score: {}'.format(score))

  def predict(self, x_predict):
    results = self.model.predict(x_predict)
    for x in results:
      #print(x)
      results = np.round(x).astype(int)
      result_a = results[0]
      result_b = results[1]

      if result_a == result_b:
        result_a = np.floor(x[0]).astype(int)
        result_b = np.ceil(x[1]).astype(int)

      if result_a == result_b:
        result_a = np.ceil(x[0]).astype(int)
        result_b = np.floor(x[1]).astype(int)

      if result_a == result_b:
        if x[0] > x[1]:
          x[0] = x[0] + 1
          result_a = np.round(x[0]).astype(int)
          result_b = np.round(x[1]).astype(int)
        else:
          x[1] = x[1] + 1
          result_a = np.round(x[0]).astype(int)
          result_b = np.round(x[1]).astype(int)



      print(result_a, result_b)
