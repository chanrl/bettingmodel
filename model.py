import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score 
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.linear_model import LogisticRegression , Ridge, Lasso, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors, datasets
import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.formula.api as smf
from glm.glm import GLM
from glm.families import Gaussian
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import re
from knn import *

data = pd.read_csv('model_data.csv').drop(columns='Unnamed: 0')
data.dropna(inplace = True)

X = np.array(data.drop(columns='Cover'))
y = np.array(data['Cover'])

def my_rmse(y_true, y_pred):
    mse = ((y_true - y_pred)**2).mean()
    return np.sqrt(mse)

def my_cross_val_score(X_data, y_data, num_folds=3):
    ''' Returns error for k-fold cross validation. '''
    kf = KFold(n_splits=num_folds)
    error = np.empty(num_folds)
    index = 0
    reg = KNeighborsRegressor()
    for train, test in kf.split(X_data):
        reg.fit(X_data[train], y_data[train])
        pred = reg.predict(X_data[test])
        error[index] = my_rmse(pred, y_data[test])
        index += 1
    return np.mean(error)

def optimal_threshold(data):
  thresholds = np.arange(100)
  best_percentage = 0
  for threshold in thresholds:
    current_percentage = data[data['Away-ATS'] > threshold].Home_win.sum() / data['Away-ATS'][data['Away-ATS'] > threshold].count()
    if current_percentage > best_percentage:
      best_threshold, best_percentage = threshold, current_percentage
  return best_threshold, best_percentage

def optimal_spread(data):
  thresholds = np.arange(50,100)
  spread_thresholds = np.arange(data['Home Spread'].min(), data['Home Spread'].max(), .5)
  best_percentage = 0
  for threshold in thresholds:
    for s_threshold in spread_thresholds:
      current_percentage = data[(data['Away-ATS'] > threshold) & (data['Home Spread'] < s_threshold)].Home_win.sum() / data['Away-ATS'][(data['Away-ATS'] > threshold) & (data['Home Spread'] < s_threshold)].count()
      if current_percentage > best_percentage:
        best_spread, best_threshold, best_percentage = s_threshold, threshold, current_percentage
  return best_spread, best_threshold, best_percentage