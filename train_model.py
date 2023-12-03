import pandas as pd
import pickle
from sklearn.neural_network import MLPRegressor 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

current_season = pd.read_pickle('data/current_season_game_logs.pkl')
last_season = pd.read_pickle('data/last_season_game_logs.pkl')
all_logs = pd.concat([current_season, last_season])
all_logs = all_logs.reset_index()
all_logs = all_logs.drop(['Game', 'Team', 'Id', 'Date'], axis=1)

# MLP shots predictor
X = all_logs.drop('Shots', axis=1)
y = all_logs['Shots']
# print(y.groupby(y).count())
# plt.hist(y)

X_train, X_valid, y_train, y_valid = train_test_split(X, y)
model = MLPRegressor(hidden_layer_sizes=(8, 6),
                     activation='relu', solver='adam', max_iter=300)
model.fit(X_train, y_train)
print('MLP Shots train score: ' + str(model.score(X_train, y_train)))
print('MLP Shots valid score: ' + str(model.score(X_valid, y_valid)))
pickle.dump(model, open('final_models/MLP_shots_predictor.sav', 'wb'))
print('\n')

# RandForest shots predictor
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
model = RandomForestRegressor(100, max_depth=4)
model.fit(X_train, y_train)
print('RF Shots train score: ' + str(model.score(X_train, y_train)))
print('RF Shots valid score: ' + str(model.score(X_valid, y_valid)))
pickle.dump(model, open('final_models/RF_shots_predictor.sav', 'wb'))
print('\n')

# Shots with correlation predictor
X = X[['FF', 'SF', 'ixG', 'iCF', 'iFF', 'iSCF', 'iHDCF']] # based on the data analysis

X_train, X_valid, y_train, y_valid = train_test_split(X, y)
model = MLPRegressor(hidden_layer_sizes=(8, 6),
                     activation='relu', solver='adam')
model.fit(X_train, y_train)
print('Correlation Shots train score: ' + str(model.score(X_train, y_train)))
print('Correlation Shots valid score: ' + str(model.score(X_valid, y_valid)))
pickle.dump(model, open('final_models/Corr_shots_predictor.sav', 'wb'))