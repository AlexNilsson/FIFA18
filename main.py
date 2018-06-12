import os

import numpy as np

from keras.callbacks import ModelCheckpoint
from keras.layers import Input, Dense, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import keras.backend as K

from utility import csv_to_dict, keep_entries, group_by_key, unique_members_from_columns

""" FILE PATHS """
PATH_TO_ROOT = os.path.dirname(os.path.abspath(__file__))
PATH_TO_DATA = os.path.join(PATH_TO_ROOT, 'data')

""" LOAD DATA """
WORLD_CUPS        = csv_to_dict(os.path.join(PATH_TO_DATA, 'WorldCups.csv')) 
WORLD_CUP_PLAYERS = csv_to_dict(os.path.join(PATH_TO_DATA, 'WorldCupPlayers.csv')) 
WORLD_CUP_MATCHES = csv_to_dict(os.path.join(PATH_TO_DATA, 'WorldCupMatches.csv'))

""" PREPROCESS DATA"""
# EXTRACT FEATURES OF INTEREST
WORLD_CUP_MATCHES = keep_entries(WORLD_CUP_MATCHES, ['Year','Home Team Name','Away Team Name','Home Team Goals','Away Team Goals'])

# EXTRACT LIST OF ALL TEAMS
ALL_TEAMS =  unique_members_from_columns(WORLD_CUP_MATCHES, ['Home Team Name', 'Away Team Name'])

# ONE HOT ENCODINGS OF ALL TEAMS
ALL_TEAMS_ENCODING = dict(zip(ALL_TEAMS, np.eye(len(ALL_TEAMS))))

# GROUP BY YEAR/CUP

WORLD_CUP_MATCHES = group_by_key(WORLD_CUP_MATCHES, 'Year')


""" SPLIT DATA INTO TRAIN/TEST SETS """
""" AND SEPARATE FEATURES FROM LABELS """
TEST_YEAR = '2014'

TEST_DATA_RAW = WORLD_CUP_MATCHES.pop(TEST_YEAR)

TRAIN_DATA_RAW = []
for x in WORLD_CUP_MATCHES:
  for m in WORLD_CUP_MATCHES[x]:
    TRAIN_DATA_RAW.append(m)


TRAIN_DATA = [[],[]]
for x in TRAIN_DATA_RAW:
  try:
    np.array([x[2],x[3]],dtype=int)

    combined = np.append(
      np.array(ALL_TEAMS_ENCODING[x[0]],dtype=int),
      np.array(ALL_TEAMS_ENCODING[x[1]],dtype=int)
    )
    TRAIN_DATA[0].append(combined)

    TRAIN_DATA[1].append(
      np.array([x[2], x[3]], dtype=int)
    )
  except: continue

TRAIN_DATA = np.array(TRAIN_DATA)


TEST_DATA = [[],[]]
for x in TEST_DATA_RAW:
  try:
    np.array([x[2],x[3]],dtype=int)

    combined = np.append(
        np.array(ALL_TEAMS_ENCODING[x[0]],dtype=int),
        np.array(ALL_TEAMS_ENCODING[x[1]],dtype=int)
    )

    TEST_DATA[0].append(combined)

    TEST_DATA[1].append(
      np.array([x[2], x[3]], dtype=int)
    )
  except: continue

TEST_DATA = np.array(TEST_DATA)

class FIFANET():
  def __init__(self):
    self.input_shape = (186,1)
    self.output_shape = (2,1)
    self.dropout_amount = 0.2
    self.optimizer = Adam(lr=0.0001)

    self.model = self.buildNetwork()
    self.model.compile(loss='mean_squared_error', optimizer=self.optimizer)

  def buildNetwork(self):
    model = Sequential()

    model.add(Dense(512, input_dim=186))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Dropout(self.dropout_amount))
    
    model.add(Dense(256))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Dropout(self.dropout_amount))

    model.add(Dense(128))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Dropout(self.dropout_amount))

    model.add(Dense(64))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Dropout(self.dropout_amount))

    model.add(Dense(32))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Dropout(self.dropout_amount))

    model.add(Dense(16))
    #model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Dropout(self.dropout_amount))

    model.add(Dense(2, activation="relu"))

    model.summary()

    return model

  def train(self, x_train, y_train, x_test, y_test, epochs=100, batch_size=128):
    print(x_train.shape)
    print(x_train[0])
    print(x_train[1])
    print(x_train[2])


    self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    score = self.model.evaluate(x_test, y_test, batch_size=batch_size)
    print('test score: {}'.format(score))


""" EXECUTE """
fifaNet = FIFANET()
fifaNet.train(
  np.random.random((1000, 186)),
  np.random.randint(2, size=(1000, 2)),
  np.random.random((1000, 186)),
  np.random.randint(2, size=(1000, 2)),
  epochs = 1000,
  batch_size = 10
)
""" fifaNet.train(
  np.array(TRAIN_DATA[0]),
  np.array(TRAIN_DATA[1]),
  np.array(TEST_DATA[0]),
  np.array(TEST_DATA[1]),
  epochs = 1000,
  batch_size = 10
) """