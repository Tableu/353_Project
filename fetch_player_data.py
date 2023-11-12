import requests
import json
import sys
import pandas as pd

def getPlayerData(player, data_dict):
    data_dict["position"].append(player["positionCode"])
    data_dict["firstName"].append(player["firstName"]["default"])
    data_dict["lastName"].append(player["lastName"]["default"])

API_URL = "https://api-web.nhle.com/v1/"
teams = ["ANA","ARI","BOS","BUF","CAR","CBJ","CGY","CHI","COL","DAL","DET","EDM","FLA","LAK","MIN","MTL","NJD","NSH","NYI","NYR","OTT","PHI","PIT","SEA","SJS","STL","TBL","TOR","VAN","VGK","WPG","WSH"]

ids = []
data_dict = {"position": [], "firstName": [], "lastName": []}

for team in teams:
    response = requests.get(API_URL + "roster/"+team+"/20232024", params={"Content-Type": "application/json"})
    data = response.json()
    #print(team + " " + str(response.status_code))
    for player in data["forwards"]:
        ids.append(player["id"])
        getPlayerData(player,data_dict)
players_frame = pd.DataFrame(data=data_dict, index=ids)
players_frame.index.name = "id"
print(players_frame)
players_frame.to_csv("player_data.csv")