import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import csv 
import sys
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv('coin_data_history_all.csv')

bnb_df = df[df['Symbol']=='BNB']

# --> if we move the 'High' column up one relative to index, and predict for range len(df)-1, we effectively pred for next mins high every time

bnb_df['High'] = bnb_df['High'].shift(-1)
bnb_df = bnb_df[:-1]

X = bnb_df[['OpenTime','Open','Low','Close','Volume','QuoteVolume','Trades','BaseAssetVolume','QuoteAssetVolume']]
y = bnb_df['High']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.35)

#SIMPLE TREE
dtree = DecisionTreeRegressor(max_depth=3)
dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)

total = 0

for i in range(1,len(bnb_df)):
	diff = abs(bnb_df.loc[i-1,'High'] - bnb_df.loc[i,'Low'])
	total += diff

avg_diff = total / len(bnb_df)

print('mean squared error for pred BNB next min High: ',mean_squared_error(y_test,predictions))
print('mean absolute error for pred BNB next min High: ',mean_absolute_error(y_test,predictions))
print('avg price variance per minute: ',avg_diff)


## Single d tree has little predictive power



