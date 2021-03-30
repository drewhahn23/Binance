import time
import requests
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pytz import timezone
import pytz
import sys
from binance.client import Client



api_key = '{key}'
secret_key = '{key}'
client = Client(api_key='{}'.format(api_key),api_secret='{}'.format(secret_key),tld='us')



base = 'https://api.binance.us'

test_connection = '/api/v3/ping'
bids = '/api/v3/depth'
recent_trades = '/api/v3/trades'
symbols = ['HNTUSD','VTHOUSD','ONEUSD','OXTUSD','UNIUSD','QTUMUSD','ICXUSD','ZENUSD','ZECUSD','ADAUSD','STORJUSD','REPUSD','VETUSD','MATICUSD','ZRXUSD','LINKUSD','KNCUSD','RVNUSD','BANDUSD','ONTUSD','NANOUSD','ZILUSD','SOLUSD','ALGOUSD','XLMUSD','HBARUSD','WAVESUSD','ENJUSD','MANAUSD']







def clean_bid_asks(df):

	bids = df[df['side']=='bids'].head(15)
	asks = df[df['side'] == 'asks'].head(15)
	
	df = pd.concat([bids,asks],axis='index',ignore_index=True)

	return df

def steepness_bid_asks(df):

	'''Need to run this every x seconds and IF return buy for over 1 min straight then buy. If sell for over 1 min then sell
	'''

	bids = df[df['side']=='bids']
	asks = df[df['side']=='asks']

	# print(bids)
	# print(asks)

	total_bid_vol = 0
	for i in range(len(bids)):
		total_bid_vol += (bids.loc[i,'price']*bids.loc[i,'quantity'])


	total_ask_vol = 0
	for j in range(len(bids),len(df),1):
		total_ask_vol += ((asks.loc[j,'price'])*(asks.loc[j,'quantity']))
		
	# print(total_bid_vol)
	# print(total_ask_vol)
	bids_to_ask = total_bid_vol/total_ask_vol
	return bids_to_ask


def trade_history_vol(df,time):
	# need to account for volume spike as well
	df['time'] = pd.to_datetime(df['time'], unit='ms')
	df['time_diff'] = curr_time-df['time']

	df = df[df['time_diff'] <= '00:05:00.000000']
	df.reset_index(drop=True,inplace=True)


	buys = df[df['isBuyerMaker']==False]
	sells = df[df['isBuyerMaker']==True]
	buys.reset_index(drop=True,inplace=True)
	sells.reset_index(drop=True,inplace=True)

	if (len(buys) > 0) and (len(sells) > 0):

		last_5_min_buy_vol = 0
		for i in range(len(buys)):
			last_5_min_buy_vol += float(buys.loc[i,'quoteQty'])


		last_5_min_sell_vol = 0
		for j in range(len(sells)):
			last_5_min_sell_vol += float(sells.loc[j,'quoteQty'])

		buy_to_sell_vol = last_5_min_buy_vol/last_5_min_sell_vol
		return buy_to_sell_vol

	else:
		if len(buys) == 0:
			return .0001
		elif len(sells) == 0:
			return 100000


def history_plus_steepness(df,time,df2):
	return_list = {}

	check1 = steepness_bid_asks(df)
	check2 = trade_history_vol(df2,time)

	return_list['bids_ask_ratio'] = check1
	return_list['buy_to_sell_vol'] = check2


	if check1 > 2 and check2 > 1.5:
		return_list['verdict'] = 'buy'
		# return print('buy')
	elif check1 < .5 and check2 < .66:
		return_list['verdict'] = 'sell'
		# return print('sell')
	else:
		return_list['verdict'] = 'hold'
		# return print('bid to ask ratio: {}, buy to sell volume in last 5 mins: {}'.format(check1,check2))

	return return_list



def acc_check(api,secret):

	# acc_info = '/api/v3/account'
	# base = 'https://api.binance.us'
	client = Client(api_key='{}'.format(api),api_secret='{}'.format(secret),tld='us')
	info = client.get_account()
	account = info['balances']
	df = pd.DataFrame(columns=['asset','amount'])
	for i in range(len(account)):

		df.loc[i] = [account[i]['asset'],account[i]['free']]
	

	return df

holdings = acc_check(api_key,secret_key)
# print(holdings[holdings['asset']=='USD'])

def total_acc_value(account):
	base = 'https://api.binance.us'
	price = '/api/v3/ticker/price'
	for i in range(len(account)):
		if account.loc[i,'asset'] == 'USD':
			account['price'] = 1
			pass
		else:
			account.loc[i,'asset'] = account.loc[i,'asset']+'USD'
			symbol = account.loc[i,'asset']
			amount = account.loc[i,'amount']
			r_price = requests.get('{}{}'.format(base,price),params=dict(symbol=symbol))
			prices = r_price.json()
			account.loc[i,'price'] = prices['price']

	account = account.rename(columns={'asset':'symbol'})

	return account

total_holding_df = total_acc_value(holdings)
# print(total_holding_df)

def acc_value_final(account):
	wallet_val = 0
	for i in range(len(account)):
		wallet_val+= (float(account.loc[i,'amount']) * float(account.loc[i,'price']))

	return wallet_val


print(acc_value_final(total_holding_df))

half_acc = acc_value_final(total_holding_df)

n = half_acc/10






final_df = pd.DataFrame()
for i in symbols:
	buy_sell = '/api/v3/order'
	base = 'https://api.binance.us'

	test_connection = '/api/v3/ping'
	bids = '/api/v3/depth'
	recent_trades = '/api/v3/trades'


	
	curr_time = datetime.utcnow()
	r_bids = requests.get('{}{}'.format(base,bids),params=dict(symbol=i))
	r_recent_trades = requests.get('{}{}'.format(base,recent_trades),params=dict(symbol=i))
	results_bids = r_bids.json()
	recent_trades = r_recent_trades.json()

	frames = {side: pd.DataFrame(data=results_bids[side], columns=['price','quantity'],dtype=float) for side in ['bids','asks']}
	df= pd.DataFrame(results_bids)
	df2 = pd.DataFrame(recent_trades)
	frames_list = [frames[side].assign(side=side) for side in frames]
	data=pd.concat(frames_list,axis='index',ignore_index=True,sort=True)
	df2 = df2.sort_values('time',ascending=False)

	df = clean_bid_asks(data)

	run = history_plus_steepness(df,curr_time,df2)
	run['symbol'] = i
	temp_df = pd.DataFrame([run])
	final_df = final_df.append(temp_df,ignore_index=True)
	# time.sleep(1)

	#initialize account value. set var to half of it
	#connect to api for orders
	#some combo of scores for check1 and check2 to maximize (similar formula to f1 score)?
	#for i in final_df choose max(f1 score)
	#buy symbol with half acc value
	#while buy open - stop searching for buys
	#if verdict = sell for over one min consecutively - sell
	#start searching for new buys again

# print(final_df)

df = final_df.merge(total_holding_df,on='symbol')

print(df)

# def buy_sell(df,trade_amount):
# 	base = 'https://api.binance.us'
# 	buy_or_sell = '/api/v3/order'

	#if up 2% if verdict != buy, sell - else hold
# 	curr_time = datetime.utcnow()

# 	for i in range(len(df)):
# 		if df.loc[i,'verdict'] == 'buy':
# 			time.sleep(60)
# 			if df.loc[i,'verdict'] == 'buy':
# 				buy_symbol = str(df.loc[i,'symbol'])
# 				quantity = trade_amount/float(df.loc[i,'price'])
# 				order = client.order_market_buy(symbol=buy_symbol,quantity=quantity)

# 		if (df.loc[i,'verdict'] == 'sell') and ((df.loc[i,'amount'] * df.loc[i,'price']) > 1):
# 			time.sleep(60)
# 			if (df.loc[i,'verdict'] == 'sell') and ((df.loc[i,'amount'] * df.loc[i,'price']) > 1):
# 				sell_symbol = str(df.loc[i,'symbol'])
# 				quantity = df.loc[i,'amount']
# 				order = client.order_market_sell(symbol=sell_symbol,quantity=quantity)


# buy_sell(df,n)




