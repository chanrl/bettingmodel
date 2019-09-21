import pandas as pd 
import numpy as np
import re
import datetime

def clean_perc(row):
    away, home = row.split("%  ")
    home = home.strip("%")
    return int(away),int(home)

def clean_names(row):
    away, home = row.split("  ")
    away = re.findall('[A-Z]\D+', away)
    home = re.findall('[A-Z]\D+', home)
    return away[0], home[0]

def away_home(df, col, new_col_name=""):
  df[f'Away{new_col_name}'] = df[col].map(lambda x: x[0])
  df[f'Home{new_col_name}'] = df[col].map(lambda x: x[1])

def pipe(filename):
  table = pd.read_csv(f'{filename}')
  table.rename(columns={ table.columns[2]: "MU", table.columns[4]: "ATS", table.columns[6]:  "ML" }, inplace = True)
  clean = table[[table.columns[2], table.columns[4], table.columns[6]]].iloc[4:].reset_index(drop=True)
  clean.ATS = clean.ATS.map(clean_perc)
  clean.ML = clean.ML.map(clean_perc)
  clean.MU = clean.MU.map(clean_names)
  away_home(clean, 'MU')
  away_home(clean, 'ATS', '-ATS')
  away_home(clean, 'ML', '-ML')
  clean['Date'] = table.iloc[:,1][3]
  clean['Date'] = pd.to_datetime(clean['Date'])
  return clean.iloc[:, 3:]