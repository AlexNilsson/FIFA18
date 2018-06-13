import os

import numpy as np

from utility import csv_to_dict, keep_entries, group_by_key, unique_members_from_columns
import config as C

from FIFANET import FIFANET

""" FILE PATHS """
PATH_TO_ROOT = os.path.dirname(os.path.abspath(__file__))
PATH_TO_DATA = os.path.join(PATH_TO_ROOT, 'data')
PATH_TO_PREDICT = os.path.join(PATH_TO_ROOT, 'predict')

""" LOAD DATA """
WORLD_CUPS        = csv_to_dict(os.path.join(PATH_TO_DATA, 'WorldCups.csv'))
WORLD_CUP_PLAYERS = csv_to_dict(os.path.join(PATH_TO_DATA, 'WorldCupPlayers.csv'))
WORLD_CUP_MATCHES = csv_to_dict(os.path.join(PATH_TO_DATA, 'WorldCupMatches.csv'))

PREDICT_MATCHES = csv_to_dict(os.path.join(PATH_TO_PREDICT, 'matches.csv'))

""" PREPROCESS DATA"""
# EXTRACT FEATURES OF INTEREST
WORLD_CUP_MATCHES = keep_entries(WORLD_CUP_MATCHES, ['Year','Home Team Name','Away Team Name','Home Team Goals','Away Team Goals'])

# EXTRACT LIST OF ALL TEAMS
ALL_TEAMS =  unique_members_from_columns(WORLD_CUP_MATCHES, ['Home Team Name', 'Away Team Name'])

# ONE HOT ENCODINGS OF ALL TEAMS
ALL_TEAMS_ENCODING = dict(zip(ALL_TEAMS, np.eye(len(ALL_TEAMS))))

# GROUP BY YEAR/CUP
WORLD_CUP_MATCHES = group_by_key(WORLD_CUP_MATCHES, 'Year')

# Decay scores to account for athletic improvements over time
N_GAMES = len(WORLD_CUP_MATCHES)
for i, year in enumerate(WORLD_CUP_MATCHES):
  scoreDecay = 1 - C.SCORE_DECAY_FACTOR * (N_GAMES-i-1)/(N_GAMES-1)
  for i, match in enumerate(WORLD_CUP_MATCHES[year]):
    try:
      match[2] = scoreDecay * int(match[2])
      match[3] = scoreDecay * int(match[2])
    except:
      # Remove matches written in the wrong format in the database
      # Some entries have home team & scores mixed up and can not be trusted :(
      del match

# Matches to Predict
temp = []
for x in PREDICT_MATCHES:
  teams = list(x.values())
  match = []

  temp.append(list(np.append(
    ALL_TEAMS_ENCODING[teams[0]],
    ALL_TEAMS_ENCODING[teams[1]]
  )))

PREDICT_MATCHES = np.array(temp, dtype=int)

""" SPLIT DATA INTO TRAIN/TEST SETS """
""" AND SEPARATE FEATURES FROM LABELS """
if C.USE_TEST_DATA:
  TEST_YEAR = '2014'
  TEST_DATA_RAW = WORLD_CUP_MATCHES.pop(TEST_YEAR)

TRAIN_DATA_RAW = []
for x in WORLD_CUP_MATCHES:
  for m in WORLD_CUP_MATCHES[x]:
    TRAIN_DATA_RAW.append(m)


TRAIN_DATA_FEATURES = []
TRAIN_DATA_LABELS = []
for x in TRAIN_DATA_RAW:
  try:
    np.array([x[2],x[3]],dtype=float)

    combined = np.append( ALL_TEAMS_ENCODING[x[0]], ALL_TEAMS_ENCODING[x[1]] )
    TRAIN_DATA_FEATURES.append(list(combined))

    TRAIN_DATA_LABELS.append(
      list(np.array([x[2], x[3]], dtype=float))
    )

  except: continue

TRAIN_DATA_FEATURES = np.array(TRAIN_DATA_FEATURES, dtype=int)
TRAIN_DATA_LABELS = np.array(TRAIN_DATA_LABELS, dtype=float)

if C.USE_TEST_DATA:
  TEST_DATA_FEATURES = []
  TEST_DATA_LABELS = []
  for x in TEST_DATA_RAW:
    try:
      np.array([x[2],x[3]],dtype=float)

      combined = np.append(ALL_TEAMS_ENCODING[x[0]], ALL_TEAMS_ENCODING[x[1]])
      TEST_DATA_FEATURES.append(list(combined))

      TEST_DATA_LABELS.append(
        list(np.array([x[2], x[3]], dtype=float))
      )
    except: continue

  TEST_DATA_FEATURES = np.array(TEST_DATA_FEATURES, dtype=int)
  TEST_DATA_LABELS = np.array(TEST_DATA_LABELS, dtype=float)

""" EXECUTE """
fifaNet = FIFANET()

if C.USE_TEST_DATA:
  fifaNet.train(
    np.array(TRAIN_DATA_FEATURES),
    np.array(TRAIN_DATA_LABELS),
    np.array(TEST_DATA_FEATURES),
    np.array(TEST_DATA_LABELS),
    epochs = C.EPOCHS,
    batch_size = C.BATCH_SIZE
  )

else:
  fifaNet.train(
    np.array(TRAIN_DATA_FEATURES),
    np.array(TRAIN_DATA_LABELS),
    np.array(TRAIN_DATA_FEATURES),
    np.array(TRAIN_DATA_LABELS),
    epochs = C.EPOCHS,
    batch_size = C.BATCH_SIZE
  )

fifaNet.predict(np.array(PREDICT_MATCHES))
