from web import *
from parse import *
import os
import time
import glob
import re
import pandas as pd 
import numpy as np

teams = ['Arizona', 'Atlanta', 'Baltimore', 'Buffalo', 'Carolina', 'Chicago', \
  'Cincinnati', 'Cleveland', 'Dallas', 'Denver', 'Detroit', 'Green Bay', 'Houston', 'Indianapolis', 'Jacksonville',\
'Kansas City', 'LA Chargers', 'LA Rams', 'Miami', 'Minnesota', 'NY Giants', 'NY Jets',\
'New England', 'New Orleans', 'Oakland', 'Philadelphia', 'Pittsburgh', 'San Francisco',\
'Seattle', 'Tampa Bay', 'Tennessee', 'Washington']

# scrape every team h2h page
# s = scrape()
# for names in teams:
#   s.oddsshark(names)

os_csv = glob.glob('data/*-os*')
os_csv.sort()

#Extracting file names into a list
odds = []
for file in os_csv:
    temp_df = pd.read_csv(file)
    odds.append(temp_df)

df = pd.concat(odds)
dates = df.Date.unique()
df['Date'] = pd.to_datetime(df['Date'])

d = {"Jan": "January", "Feb": "February", "Sep":"September", "Oct":"October", "Nov": "November", "Dec": "December"}
for idx in range(len(dates)):
  dates[idx] = dates[idx].split(" ")
  dates[idx][1] = dates[idx][1].strip(',')
  if len(dates[idx][1]) == 1:
    dates[idx][1] = f'0{dates[idx][1]}'
  if dates[idx][0] in d.keys():
    dates[idx][0] = d[dates[idx][0]]

# scrape every date for nfl games played for the last 3 years
# s = scrape()
# for date in dates:
#   s.sportsplays(date[0], date[1], date[2])

dates_csv = []
sp_csv = glob.glob('data/*-sp*')
sp_csv.sort()
for filename in sp_csv:
    temp_df = pipe(filename)
    dates_csv.append(temp_df)

#sp_df = pd.concat(dates_csv)

#home spread + home score > away score? = home win

# df['Home_win'] = 0 
# df = df.reset_index(drop=True)
# df.drop(columns='Unnamed: 0', inplace = True)
# for idx in range(len(df)):
#   if (df.iloc[:,2][idx] < df.iloc[:,4][idx] + df.iloc[:,6][idx]) == True:
#     df.Home_win[idx] = 1
#   elif (df.iloc[:,2][idx] == df.iloc[:,4][idx] + df.iloc[:,6][idx]) == True:
#     df.Home_win[idx] = float('NaN')
#   else:
#     df.Home_win[idx] = 0

sp_df = pd.read_csv('sp_df')
od_df = pd.read_csv('oddsshark.csv')

team_names = np.sort(sp_df.Away.unique())
team_names_2 = np.sort(od_df.Away.unique())

d = {}
for init, names in zip(team_names_2, team_names):
  d[names] = init

sp_df['Init'] = sp_df.Home.map(lambda x: d[x])
od_df['Init'] = od_df['Home']
temp = pd.merge(sp_df, od_df, on=['Init','Date'])
merged = temp[['Date', 'Init', 'Home Spread', 'Home_win', 'Away-ATS', 'Home-ATS', 'Away-ML', 'Home-ML']].drop_duplicates()

#model
#pub betting 70% on Away. 

model_df = merged[['Home Spread', 'Home_win', 'Away-ATS']].reset_index(drop = True)
model_df = model_df.rename(columns = {'Home_win': 'Cover'})
model_df.to_csv('model_data.csv')