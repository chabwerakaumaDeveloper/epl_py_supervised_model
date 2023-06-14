from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def soccer_pred(task,test_features):
  #load data
  data = pd.read_csv('') #insert path of a .csv file containing epl season data.
  X = data[['HomeTeam','AwayTeam']]
  if task == 1:
    #TC total corners.
    data['TC'] = data['HC'] + data['AC']
    y = data['TC']
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn.fit(X,y)
    result = knn.predict(test_features)
    print(result)
  elif task == 2:
    #TBP total booking points.
    data['TBP'] = (data['HY'] + data['AY'])
    X = data[['HomeTeam','AwayTeam','Referee']]
    y = data['TBP']
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn.fit(X,y)
    result = knn.predict(test_features)
    print(result)
  elif task == 3:
    #TGHT total goals half time.
    data['TGHT'] = data['HTAG'] + data['HTHG'] 
    y = data['TGHT']
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn.fit(X,y)
    result = knn.predict(test_features)
    print(result)
  elif task == 4:
    #TGFT total goals full time.
    data['TGFT'] = data['FTAG'] + data['FTHG'] 
    y = data['TGFT']
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn.fit(X,y)
    result = knn.predict(test_features)
    print(result)
  elif task == 5:
    #First half winner.
    y = data['HTR']
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X,y)
    result = knn.predict(test_features)
    print(result)
  elif task == 6:
    #Fulltime winner.
    y = data['FTR']
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X,y)
    result = knn.predict(test_features)
    print(result)
  elif task == 7:
    #Half time both score.
    y = data['HTGG']
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X,y)
    result = knn.predict(test_features)
    print(result)
  elif task == 8:
    #Second time both score.
    y = data['FTGG']
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X,y)
    result = knn.predict(test_features)
    print(result)
  elif task == 9:
    #Goals even or odd.
    y = data['GEO']
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn.fit(X,y)
    result = knn.predict(test_features)
    print(result)

soccer_pred(4,[[10,4]])