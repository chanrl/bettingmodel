import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, roc_curve
from sklearn.linear_model import LogisticRegression , Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler
import scipy.stats as scs
import statsmodels.api as sm
import statsmodels.formula.api as smf
from glm.glm import GLM
from glm.families import Gaussian
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import re

data = pd.read_csv('model_data.csv').drop(columns='Unnamed: 0')
data.dropna(inplace = True)

def my_rmse(y_true, y_pred):
    mse = ((y_true - y_pred)**2).mean()
    return np.sqrt(mse)

def my_cross_val_scores(X_data, y_data, num_folds=3):
    ''' Returns error for k-fold cross validation. '''
    kf = KFold(n_splits=num_folds)
    train_error = np.empty(num_folds)
    test_error = np.empty(num_folds)
    index = 0
    linear = LinearRegression()
    for train, test in kf.split(X_data):
        linear.fit(X_data[train], y_data[train])
        pred_train = linear.predict(X_data[train])
        pred_test = linear.predict(X_data[test])
        train_error[index] = my_rmse(pred_train, y_data[train])
        test_error[index] = my_rmse(pred_test, y_data[test])
        index += 1
    return np.mean(test_error), np.mean(train_error)

#my_cross_val_scores(data, data.Home_win, 5)

def optimal_threshold(data):
  thresholds = np.arange(100)
  best_percentage = 0
  for threshold in thresholds:
    current_percentage = data[data['Away-ATS'] > threshold].Home_win.sum() / data['Away-ATS'][data['Away-ATS'] > threshold].count()
    if current_percentage > best_percentage:
      best_threshold, best_percentage = threshold, current_percentage
  return best_threshold, best_percentage

optimal_threshold(data)

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

s, t,p = optimal_spread(data)

#when pub bet > 50 on away , home spread -5 or more, 6/20, 70% wim rate

#pub bet > 70 on away , home spread +3, 69% win rate