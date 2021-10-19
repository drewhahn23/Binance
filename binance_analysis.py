import pandas as pd 
import numpy as np 
import matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns 
import csv 
import datetime
from dateutil import parser
from functools import reduce
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn import metrics

btc = pd.read_csv('init_binance_data_btc.csv')
hbar_ask = pd.read_csv('init_binance_data_asks_hbar.csv')
hbar_bid = pd.read_csv('init_binance_data_bids_hbar.csv')
trades = pd.read_csv('init_binance_data_trades_hbar.csv')


# print(trades)
# print(hbar_ask)
hbar_bid.columns = ['datetime','bid_price','bid_quantity']

hbar_ask.columns = ['datetime','ask_price','ask_quantity']

btc.columns = ['datetime','btc_price','btc_quantity','btc_trade_val']

trades.columns = ['market_buy_sell','datetime','trade_price','trade_quantity']


# print(len(hbar_bid)) #33631
# print(len(hbar_ask)) #34007
# print(len(btc)) #7225
# print(len(trades)) #12598

# hbar_bid = hbar_bid.resample('m',on='datetime').agg({'bid_price':'mean','bid_quantity':'sum'})

hbar_bid.reset_index(inplace=True,drop=True)

hbar_bid['datetime'] = pd.to_datetime(hbar_bid['datetime'],format='%Y-%m-%d %H:%M:%S.%f')

hbar_bid['datetime'] = hbar_bid['datetime'].astype('datetime64[m]')

hbar_ask.reset_index(inplace=True,drop=True)

hbar_ask['datetime'] = pd.to_datetime(hbar_ask['datetime'],format='%Y-%m-%d %H:%M:%S.%f')

hbar_ask['datetime'] = hbar_ask['datetime'].astype('datetime64[m]')

btc.reset_index(inplace=True,drop=True)

btc['datetime'] = pd.to_datetime(btc['datetime'],format='%Y-%m-%d %H:%M:%S.%f')

btc['datetime'] = btc['datetime'].astype('datetime64[m]')

trades.reset_index(inplace=True,drop=True)

trades['datetime'] = pd.to_datetime(trades['datetime'],format='%Y-%m-%d %H:%M:%S.%f')

trades['datetime'] = trades['datetime'].astype('datetime64[m]')


agg1m_bid = hbar_bid.groupby('datetime').mean()
# print(agg1m_bid)

agg1m_ask = hbar_ask.groupby('datetime').mean()
# print(agg1m_ask)

agg_btc = btc.groupby('datetime').mean()
# print(agg_btc)

agg_trades = trades.groupby('datetime').mean()
# print(agg_trades)


dfs = [agg1m_bid,agg1m_ask,agg_btc,agg_trades]

final_df = reduce(lambda left,right: pd.merge(left,right,on='datetime'),dfs)
final_df.reset_index(inplace=True,drop=False)

# final_df['BSH'] = 

# print(final_df)

final_df.to_csv('binance_data_final.csv',sep=',',index=False)


'''
dates = matplotlib.dates.date2num(final_df['datetime'])
plt.plot_date(dates,final_df['trade_price'])
plt.show()

X = final_df.drop(['trade_price','datetime'],axis=1)
y = final_df['trade_price']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3)


lm = LinearRegression()

lm.fit(X_train,y_train)

preds = lm.predict(X_test)

# print(len(X_test.columns))

# single = np.array([.3140,1206.1,.3147,3636.9,58778.57,.025,3000,.6,1000]).reshape(1,-1)
# new = lm.predict(single)
#predicted .3193 when price was .3140
# print(new)

# print(preds,X_test)
# plt.scatter(y_test,dates,c='red')
# plt.scatter(preds,c='blue')
# plt.show()

print(np.sqrt(metrics.mean_squared_error(y_test,preds)))


'''