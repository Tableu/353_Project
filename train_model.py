import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor

current_season = pd.read_pickle('data/current_season_game_logs.pkl')
last_season = pd.read_pickle('data/last_season_game_logs.pkl')
all_logs = pd.concat([current_season, last_season])
# all_logs = all_logs.drop(['Game', 'Team'], axis=1)
print(all_logs)

