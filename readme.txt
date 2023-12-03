Steps to run:

1: Fetching and cleaning data
python3 fetch_game_log_data.py data/game_log_data.pkl #shouldn't need to run this since all data is already in game_log_data.pkl
python3 fetch_player_data.py
python3 clean_data.py data/game_log_data.pkl
2: Data Analysis
python3 data_analysis.py data/last_season_game_logs_all.pkl
3: Training Models
4: Rankings and Plots
python3 plot_top_rankings.py output/ranking_analysis.csv
python3 rankings.py
python3 shots_predictor.py