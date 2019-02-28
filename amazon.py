import pandas as pd
import csv
import numpy as np
import math
from sklearn import preprocessing,cross_validation, svm
from sklearn.linear_model import LinearRegression

filename = 'AMZN.csv'
data = pd.read_csv(filename)
data['HL_Pct'] = (data['High'] - data['Adj Close'])/ data['Adj Close'] * 100.0
data['PCT_change'] = (data['Adj Close'] - data['Open'])/data['Open'] * 100.0

data = data[['Adj Close', 'HL_Pct','PCT_change','Volume']]
forecast_col = 'Adj Close'
data.fillna(-99999, inplace=True) 

forecast_out =int(math.ceil(0.01*len(data)))
print(forecast_out)

data['label'] = data[forecast_col].shift(-forecast_out)
data.dropna(inplace= True)
print(data.head())

X = np.array(data.drop(['label'],1))
y = np.array(data['label'])
x = preprocessing.scale(X)
y=np.array(data['label'])


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

classifier = LinearRegression()
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)
print(accuracy) 
