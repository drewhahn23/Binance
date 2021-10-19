import lightgbm as lgb 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

df = pd.read_csv('binance_data_final_edit_1.csv')
one_hot_bsh = pd.get_dummies(df[['BSH']],drop_first=False).values
print(one_hot_bsh)
final_df = pd.concat([df,one_hot_bsh],axis=1)
final_df.drop(['BSH'],axis=1,inplace=True)

X = final_df[['bid_price','bid_quantity','ask_price','ask_quantity','btc_price','btc_quantity','btc_trade_val','market_buy_sell','trade_price','trade_quantity']]
y = final_df[['BSH_Buy','BSH_Hold','BSH_Sell']]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3)

d_train = lgb.Dataset(X_train,label=y_train)
print(d_train)

params = {
	'task':'train',
	'num_class':3,
	'max_depth': 5,
	'num_leaves':25,
	'objective': 'multiclass',
	'learning_rate': [.01,.1,.15],
	'metric':'multi_logloss',
	'boosting_type':'gbdt'
}


clf = lgb.train(params,d_train,100)
y_pred_1 = clf.predict(X_test)
print(y_pred_1)
# model = lgb.LGBMClassifier(**params)

# model.fit(X_train,y_train)

# preds = model.predict(X_test)

# predictions = []

# for x in preds:
# 	predictions.append(np.argmax(x))

# print(predictions)
# print(classification_report(y_test,predictions))
# print(confusion_matrix(y_test,predictions))