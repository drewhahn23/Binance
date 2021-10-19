import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import csv 
import sys
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
# from tensorflow.keras.models import Sequential



final_df = pd.read_csv('coin_data_history_all.csv')
# final_df['symbol'] = final_df.index

#X = final_df[['OpenTime','Open','High','Low','Close','Volume','CloseTime','QuoteVolume','Trades','BaseAssetVolume','QuoteAssetVolume']]

# bnb_df = final_df[final_df['Symbol']=='BNB']

symbol_list = final_df['Symbol'].unique()
list_of_dfs = []

for i in range(len(symbol_list)):
	symbol_list[i] = final_df[final_df['Symbol']==symbol_list[i]]

	list_of_dfs.append(symbol_list[i])

# print(list_of_dfs)

#linear model of high prices - not much to gather from graphing as not graphed relative to time data. FURTHERMORE
#for each minute we are assuming we have entire minutes trading statistics and use those to predict - this is obviously not true
# need to be predicting for NEXT minute - need to adjust entire setup of data. can do so not-in-place for now but eventually may want to rewrite entire CSV
# if predictive power is deemed significant

for i in range(len(list_of_dfs)):
	print(list_of_dfs[i])
	try:

		list_of_dfs[i] = list_of_dfs[i].drop('CloseTime',axis=1)
		print(list_of_dfs[i].loc[0,'Symbol'])

		X = list_of_dfs[i][['OpenTime','Open','Low','Close','Volume','QuoteVolume','Trades','BaseAssetVolume','QuoteAssetVolume']]
		y = list_of_dfs[i]['High']

		X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.4)

		lm = LinearRegression()
		lm.fit(X_train,y_train)

		predictions = lm.predict(X_test)


		total = 0
		for j in range(len(list_of_dfs[i])):
			diff = abs(list_of_dfs[i].loc[j,'High'] - list_of_dfs[i].loc[j,'Low'])
			total += diff

		avg_diff = total / len(list_of_dfs[i])

		print('mean squared error: ',mean_squared_error(y_test,predictions))
		print('avg diff between high and low: ', avg_diff)
		print('mean absolute error: ', mean_absolute_error(y_test,predictions))

	except Exception as e:
		print(e)





# print(bnb_df)
# print(final_df)