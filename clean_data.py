import sys
import pandas as pd
#splits the game logs by seasons and removes players who played < 20 games last season

data = pd.read_pickle(sys.argv[1])
data.index = data.index.set_levels([data.index.levels[0], pd.to_datetime(data.index.levels[1])])
last_season = data[data.index.get_level_values('Date') < "2023-6-20"]
current_season = data[data.index.get_level_values('Date') > "2023-6-20"]

game_counts = last_season.groupby(['Id']).size().to_frame('size')
game_counts = game_counts[game_counts['size'] <= 20]

last_season = last_season.drop(index = game_counts.index, level='Id')
current_season = current_season.loc[last_season.index.get_level_values('Id'),:]

last_season.to_pickle("data/last_season_game_logs.pkl")
current_season.to_pickle("data/current_season_game_logs.pkl")