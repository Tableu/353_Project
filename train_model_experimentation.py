import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sp
import pickle
from sklearn.neural_network import MLPRegressor 
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import classification_report

current_season = pd.read_pickle('data/current_season_game_logs.pkl')
last_season = pd.read_pickle('data/last_season_game_logs.pkl')
all_logs = pd.concat([current_season, last_season])
all_logs = all_logs.reset_index()
all_logs = all_logs.drop(['Game', 'Team', 'Id', 'Date'], axis=1)
# print(all_logs.groupby(all_logs['Shots']).count())

# MLP total points predictor
X = all_logs.drop('Total Points', axis=1)
y = all_logs['Total Points']

X_train, X_valid, y_train, y_valid = train_test_split(X, y)
model = MLPRegressor(hidden_layer_sizes=(8, 6),
                     activation='relu', solver='adam', max_iter=300)
model.fit(X_train, y_train)
print('MLP Points train score: ' + str(model.score(X_train, y_train)))
print('MLP Points valid score: ' + str(model.score(X_valid, y_valid)))
pickle.dump(model, open('test_models/MLP_total_points_predictor.sav', 'wb'))
print('\n')
# Experimental results suggest this is not a good model. The training and valiation
# scores are really, good, but based on the distribution of the Total Points, it is
# not a good predictor. Balancing the data reduces results in too much data loss.
# predicting Shots on the other hand may be better because the data is better distributed
# and the shots measures how many time the player tried to score, which would sugges that
# the higher the shots, the more the chances of scoring and winning games (performing better)


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
pickle.dump(model, open('test_models/MLP_shots_predictor.sav', 'wb'))
print('\n')
# this is a good predictor and should be used to predict how many potential shots a 
# player will make in conjunction with other test_models. The training and validation scores
# are not too bad (around 0.82) and hence works well


# KNN shots predictor
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
model = KNeighborsRegressor(5)
model.fit(X_train, y_train)
print('KNN Shots train score: ' + str(model.score(X_train, y_train)))
print('KNN Shots valid score: ' + str(model.score(X_valid, y_valid)))
pickle.dump(model, open('test_models/KNN_shots_predictor.sav', 'wb'))
print('\n')
# this is a bad predictor, too many dimensions and data distribution for some dimensions
# is 'interesting' to say the least. Bad validation score (0.47)


# RandForest shots predictor
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
model = RandomForestRegressor(30, max_depth=4)
model.fit(X_train, y_train)
print('RF Shots train score: ' + str(model.score(X_train, y_train)))
print('RF Shots valid score: ' + str(model.score(X_valid, y_valid)))
pickle.dump(model, open('test_models/RF_shots_predictor.sav', 'wb'))
print('\n')
# This model does not perform as well as the MLP with shots but does pretty good in terms
# of balancing the predictions; validation score (0.79)


# PCA MLP shots predictor
X_train, X_valid, y_train, y_valid = train_test_split(X, y)
model = make_pipeline(
    PCA(8), # same number of dimensions as the most the ones with moderately strong correlation
    MLPRegressor(hidden_layer_sizes=(8, 6),
                     activation='relu', solver='adam')
)
model.fit(X_train, y_train)
print('PCA Shots train score: ' + str(model.score(X_train, y_train)))
print('PCA Shots valid score: ' + str(model.score(X_valid, y_valid)))
pickle.dump(model, open('test_models/PCA_MLP_shots_predictor.sav', 'wb'))
print('\n')
# Since there is a high number of dimensions, extracting the ones with the highest variance 
# seemed like a good idea, however, due to the 'interesting' distribution of data, the model
# was not good: validation score (0.43)


# Shots with correlation predictor
X = X[['FF', 'SF', 'ixG', 'iCF', 'iCF', 'iFF', 'iSCF', 'iHDCF']] # based on the data analysis

X_train, X_valid, y_train, y_valid = train_test_split(X, y)
model = MLPRegressor(hidden_layer_sizes=(8, 6),
                     activation='relu', solver='adam')
model.fit(X_train, y_train)
print('Correlation Shots train score: ' + str(model.score(X_train, y_train)))
print('Correlation Shots valid score: ' + str(model.score(X_valid, y_valid)))
pickle.dump(model, open('test_models/Corr_shots_predictor.sav', 'wb'))
# After looking at the data analysis and plots, these columns were the ones that showed some sort
# of a linear relationship and hence only using these to predict shots seems like a good idea
# the resulting scores were very good as well; validation score (0.84)
