import requests
import json
import pandas as pd
import time
import random
import sys

def getPlayerData(player, data_dict):
    data_dict["position"].append(player["positionCode"])
    data_dict["firstName"].append(player["firstName"]["default"])
    data_dict["lastName"].append(player["lastName"]["default"])

def scrapeNaturalStatTrick(id):
    url = 'https://www.naturalstattrick.com/playerreport.php?fromseason=20222023&thruseason=20232024&stype=2&sit=5v5&stdoi={}&rate=n&v=g&playerid={}'.format(
    'oi',id) 
    oi_df = None
    try:
        oi_df = pd.read_html(url, na_values=["-"])[0]
    except:
        return None
    #print(oi_df)
    time.sleep(random.randint(5,10))
    url = 'https://www.naturalstattrick.com/playerreport.php?fromseason=20222023&thruseason=20232024&stype=2&sit=5v5&stdoi={}&rate=n&v=g&playerid={}'.format(
    'std',id) 
    std_df = None
    try:
        std_df = pd.read_html(url, na_values=["-"])[0]
    except:
        return None
    #print(std_df)

    df = pd.merge(oi_df,std_df,on='Game', suffixes=('','_y'))
    df.drop(df.filter(regex='_y$').columns, axis=1, inplace=True)
    df['Date'] = df['Game'].str[:-11]
    df['Id'] = str(id)
    df.set_index(['Id', 'Date'], inplace=True)
    #print(df.columns)
    #print(df)
    time.sleep(random.randint(5,10))
    return df

API_URL = "https://api-web.nhle.com/v1/"
teams = ["ANA","ARI","BOS","BUF","CAR","CBJ","CGY","CHI","COL","DAL","DET","EDM","FLA","LAK","MIN","MTL","NJD","NSH","NYI","NYR","OTT","PHI","PIT","SEA","SJS","STL","TBL","TOR","VAN","VGK","WPG","WSH"]

ids = []
data_dict = {"position": [], "firstName": [], "lastName": []}

saved_data = None
if(len(sys.argv) == 2):
    saved_data = pd.read_pickle(sys.argv[1])

for team in teams:
    response = requests.get(API_URL + "roster/"+team+"/20232024", params={"Content-Type": "application/json"})
    data = response.json()
    #print(team + " " + str(response.status_code))
    for player in data["forwards"]:
        ids.append(player["id"])
        #getPlayerData(player,data_dict)
    '''for player in data["defensemen"]:
        ids.append(player["id"])
        getPlayerData(player, data_dict)
    for player in data["goalies"]:
        ids.append(player["id"])
        getPlayerData(player, data_dict)'''

df_list = []
i = 0

if(saved_data is not None):
    df_list.append(saved_data)
    saved_ids = saved_data.index.get_level_values(0).drop_duplicates()
    for id in saved_ids:
        ids.remove(int(id))
print("Remaining: " + str(len(ids)))
for id in ids:
    df = scrapeNaturalStatTrick(id)
    if(df is None):
        break
    df_list.append(df)
    print(i)
    i += 1

df = pd.concat(df_list)
print(df)
df.to_pickle("game_log_data.pkl")