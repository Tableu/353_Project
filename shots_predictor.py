import numpy as np
import pandas as pd
import pickle

def main():
    players = pd.read_csv('data/player_data.csv')
    current_season = pd.read_pickle('data/current_season_game_logs.pkl')
    last_season = pd.read_pickle('data/last_season_game_logs.pkl')
    games = pd.concat([current_season, last_season])
    games = games.reset_index()
    games['Id'] = games['Id'].astype(np.int64)
    model = [0, 0, 0]
    model[0] = pickle.load(open('final_models/MLP_shots_predictor.sav', 'rb'))
    model[1] = pickle.load(open('final_models/RF_shots_predictor.sav', 'rb'))
    model[2] = pickle.load(open('final_models/Corr_shots_predictor.sav', 'rb'))
    # print(model[0])
    value = input("Enter the player name (first and last):\n")
    # if isinstance(value, str):
    name = value.split(' ')
    assert len(name) == 2, f"Player name invalid"
    player = players[players['firstName'] == name[0]]
    player = player[player['lastName'] == name[1]]
    if(len(player) == 0):
        print("Player not found, try again")
        exit()
    player_games = games[games['Id'] == player.iloc[0]['id']]
    player_games = player_games.sort_values('Date', axis=0, ascending=False)
    player_games = player_games.drop(player_games.index.to_list()[3:], axis=0)
    player_games = player_games.drop(['Game', 'Team', 'Date', 'Shots'], axis=1)
    player_games = player_games.groupby('Id').mean()
    player_games = player_games.reset_index()
    player_games = player_games.drop('Id', axis=1)
    prediction = [0, 0, 0]
    if(player_games.shape[0] != 0):
        prediction = [model[0].predict(player_games)[0], model[1].predict(player_games)[0], 0]
    player_games = player_games[['FF', 'SF', 'ixG', 'iCF', 'iFF', 'iSCF', 'iHDCF']]
    if(player_games.shape[0] != 0):
        prediction[2] = model[2].predict(player_games)[0]
    prediction = np.mean(prediction)
    print('Expected shots in the next game: ' + str(prediction))

main() 