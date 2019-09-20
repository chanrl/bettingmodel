from web import * 
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
os = []
for file in os_csv:
    temp_df = pd.read_csv(file)
    os.append(temp_df)

df = pd.concat(os)
dates = df.Date.unique()

d = {"Jan": "January", "Feb": "February", "Sep":"September", "Oct":"October", "Nov": "November", "Dec": "December"}
for idx in range(len(dates)):
  dates[idx] = dates[idx].split(" ")
  dates[idx][1] = dates[idx][1].strip(',')
  if len(dates[idx][1]) == 1:
    dates[idx][1] = f'0{dates[idx][1]}'
  if dates[idx][0] in d.keys():
    dates[idx][0] = d[dates[idx][0]]

# scrape every date for nfl games played for the last 3 years
s = scrape()
for date in dates:
  s.sportsplays(date[0], date[1], date[2])

# s.quit()

#l[2].strip(",")
# sp = []
# sp_csv = glob.glob('data/*-sp*')
# sp_csv.sort()
# for file in sp_csv:
#     path, filename = os.path.split(file)
#     #year = re.findall('\d\d\d\d', filename)[0]
#     temp_df = pd.read_csv(file)
#     #temp_df['file_year'] = year
#     sp.append(temp_df)