import time
import requests
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime,timedelta
from pytz import timezone
import pytz
import sys
# from binance.client import Client
import math
import json
import hashlib
import hmac
import urllib.parse
import urllib.request 
# import settings
import ssl
from binance import Client, ThreadedWebsocketManager
from binance.enums import *

try:
	_create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
	# Legacy Python that doesn't verify HTTPS certificates by default
	pass
else:
	# Handle target environment that doesn't support HTTPS verification
	ssl._create_default_https_context = _create_unverified_https_context






client = Client(api_key='{}'.format(api_key),api_secret='{}'.format(secret_key),tld='us')



base = 'https://api.binance.us'

test_connection = '/api/v3/ping'
bids = '/api/v3/depth'
recent_trades = '/api/v3/trades'
avg_price = '/api/v3/avgPrice'
klines = '/api/v3/klines'
symbols = ['HNTUSD','VTHOUSD','ONEUSD','OXTUSD','UNIUSD','QTUMUSD','ICXUSD','ZENUSD','ZECUSD','ADAUSD','STORJUSD','REPUSD','VETUSD','MATICUSD','ZRXUSD','LINKUSD','KNCUSD','RVNUSD','BANDUSD','ONTUSD','NANOUSD','ZILUSD','SOLUSD','ALGOUSD','XLMUSD','HBARUSD','WAVESUSD','ENJUSD','MANAUSD']
sell_order = '/api/v3/order'
buy_order = '/api/v3/order'
# hashedsig = hashlib.sha256(secret_key)

def sigmoid(x):
	return 1/(1+math.exp(-x))

# def hand_made_activation_func():
# 	return_list['bids_ask_ratio'] = check1
# 	return_list['buy_to_sell_vol_5_min'] = check2
# 	return_list['buy_to_sell_vol_15_min'] = check3

# 	score = sigmoid(check1) * math.tanh(check3) * math.tanh(check2/3)




def clean_bid_asks(df):
	##initially tried (head(15))
	bids = df[df['side']=='bids'].head(25)
	asks = df[df['side'] == 'asks'].head(25)
	
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
	df['time_diff'] = time-df['time']
	## lets change this to have a 15 min vol checker, then a 5 min checker -> then can use both to determine shifts by implementing both into 3 pronged f_score

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

		buy_to_sell_vol_5_min = last_5_min_buy_vol/last_5_min_sell_vol
		return buy_to_sell_vol_5_min

	else:
		if len(buys) == 0:
			return .1
		elif len(sells) == 0:
			return 10

def trade_vol_15_min(df,time):
	df['time'] = pd.to_datetime(df['time'], unit='ms')
	df['time_diff'] = time-df['time']
	## lets change this to have a 15 min vol checker, then a 5 min checker -> then can use both to determine shifts by implementing both into 3 pronged f_score

	df = df[df['time_diff'] <= '00:15:00.000000']
	df.reset_index(drop=True,inplace=True)


	buys = df[df['isBuyerMaker']==False]
	sells = df[df['isBuyerMaker']==True]
	buys.reset_index(drop=True,inplace=True)
	sells.reset_index(drop=True,inplace=True)

	if (len(buys) > 0) and (len(sells) > 0):

		last_15_min_buy_vol = 0
		for i in range(len(buys)):
			last_15_min_buy_vol += float(buys.loc[i,'quoteQty'])


		last_15_min_sell_vol = 0
		for j in range(len(sells)):
			last_15_min_sell_vol += float(sells.loc[j,'quoteQty'])

		buy_to_sell_vol_15_min = last_15_min_buy_vol/last_15_min_sell_vol
		return buy_to_sell_vol_15_min

	else:
		if len(buys) == 0:
			return .1
		elif len(sells) == 0:
			return 10

	### fix returns for else statement
	### in history_plus_steepness, add checks for buys and sells (i.e. make sure bid_to_ask > 1.5 for buy < .66 for sell etc - narrow down buys and sells w more qualifiers)



def history_plus_steepness(df,time,df2,df3):
	return_list = {}

	check1 = steepness_bid_asks(df)
	check2 = trade_history_vol(df2,time)
	check3 = trade_vol_15_min(df3,time)
	#check3 used to take time1, but time1 was just the same thing as curr_time i believe?

	return_list['bids_ask_ratio'] = check1
	return_list['buy_to_sell_vol_5_min'] = check2
	return_list['buy_to_sell_vol_15_min'] = check3

	# f_score = 0
	# ###2 * val1
	# f_score = (2 * (1.5 * check1) * check2 * (2 * check3))/(check1+check2+check3)
	score = sigmoid(check1) * math.tanh(check3) * math.tanh(check2/3)
	##2, 2, 2 as checks returns .8. .5,.5,.5 returns .08
	if score > .88:
		return_list['verdict'] = 'buy'
		return_list['score'] = score
	elif score < .05:
		##init lower bound value = .57
		return_list['verdict'] = 'sell'
		return_list['score'] = score
	else:
		return_list['verdict'] = 'hold'
		return_list['score'] = score
	# if check1 > 2 and check2 > 1.5:
	# 	return_list['verdict'] = 'buy'
	# 	# return print('buy')
	# elif check1 < .5 and check2 < .66:
	# 	return_list['verdict'] = 'sell'
	# 	# return print('sell')
	# else:
	# 	return_list['verdict'] = 'hold'
		# return print('bid to ask ratio: {}, buy to sell volume in last 5 mins: {}'.format(check1,check2))

	return return_list






##how do we weight all vars for new score *call factor score*
## (2 * (vol_15_min * (2*vol_5_min) * (1.5 * steepness_bid_ask))) / (vol_15_min + vol_5_min + steepness_bid_ask)
##upper limit = 2 * (1.5*3*3)/(1.5+1.5+2) = 5.4 -> 5
## lower limit = 2 * (.66 * (2*.66) * (1.5*.5))/(.66+.66+.5) =1.3068/1.82 = .718 -> lower limit .5


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


# print(holdings[holdings['asset']=='USD'])
# print(holdings)

def total_acc_value(account):
	base = 'https://api.binance.us'
	price = '/api/v3/ticker/price'
	price_df = pd.DataFrame()
	price_list = []
	for i in range(len(account)):
		# if account.loc[i,'asset'] == 'USD':
		# 	account['price'] = 1
		# 	pass
		# else:
		account.loc[i,'asset'] = account.loc[i,'asset']+'USD'
		symbol = account.loc[i,'asset']
		amount = account.loc[i,'amount']
			# df = pd.DataFrame()
		r_price = requests.get('{}{}'.format(base,price),params=dict(symbol=symbol))

		prices = r_price.json()
		price_list.append(prices)
			# print(prices)
			# account['price'] = prices['price']
			# df = df.append(prices,ignore_index=True)
			# price_df = price_df.append(df)
			# account.loc[i,'price'] = prices['price']
			# price_df = pd.DataFrame(prices)


	for i in range(len(account)-1):
		if 'price' in price_list[i].keys():
			account.loc[i,'price'] = price_list[i]['price']	
		else:
			pass

	account = account.rename(columns={'asset':'symbol'})

	return account


# print(total_holding_df)

def acc_value_final(account):
	wallet_val = 0
	for i in range(len(account)):
		wallet_val+= (float(account.loc[i,'amount']) * float(account.loc[i,'price']))

	return wallet_val




def parse_kline_data(list1):

	last_min = list1
	var_list = ['open_time','open','high','low','close','volume','close_time','quote_asset_volume','num_trades','taker_buy_base_asset_vol','taker_buy_quote_asset_vol','ignore']

	kline_dict = dict(zip(var_list,last_min))
	# print(kline_dict)
	return kline_dict




def comp_kline_time_diff(df_1m,df_1h):

	for i in range(len(df_1h)):
		sixty_m_vol = df_1h.loc[i,'volume']
		minute_vol = df_1m.loc[i,'volume']

		df_1m.loc[i,'60m_vol'] = sixty_m_vol

		if float(minute_vol) >= float(sixty_m_vol) / 15:
			df_1m['volume_spike'] = 'yes'
		else:
			df_1m['volume_spike'] = 'no'

	return df_1m

def vol_spike_override(df):
	for i in range(len(df)):
		if df.loc[i,'score'] > .5 and df.loc[i,'volume_spike'] == 'yes':
			df['verdict'] = 'buy'
		else:
			pass
	
	return df
# df = df.join(kline_data_df_1h,how='outer')
#do we want to add 15m vol as well?

def manage_holdings(return_df,coin_held_check,price_buy):
	# assign initial value of purchase to var
	# continue checking value on every run
	# if coin rating falls below buy, check f_score
	# if f_score > 5 continue holding
	# elif f_score < 5 and % gain greater than 5 then sell
	# elif f_score < 5 and % gain less than 5 then hold
	# elif f_score < 1 --> sell regardless of price
	buy_rating = []
	# value_list = []
	
	symbol_to_buy = ''
	#### I AM LOOPING AND RESETTING BOUGHT AS FALSE EVERY TIME - NOT PROPER LOOP ----> NEVER LOOKS TO SELL
	
	if coin_held_check != True:
		for i in range(len(return_df)):
			append_dict = {}
			if return_df.loc[i,'verdict'] == 'buy':
				append_dict['symbol'] = return_df.loc[i,'symbol']
				append_dict['score'] = return_df.loc[i,'score']
				# value_list.append(append_dict['score'])
				buy_rating.append(append_dict)
		print(buy_rating)
		holdings = acc_check(api_key,secret_key)
		total_holding_df = total_acc_value(holdings)
		# print(max(value_list))
		if len(buy_rating) > 0:
			seq = [x['score'] for x in buy_rating]
			max_value = max(seq)
			# print(max_value)
			buy_dict = next(item for item in buy_rating if item['score'] == max_value)
			symbol_to_buy += str(buy_dict['symbol'])
			row = return_df[return_df['symbol'] == symbol_to_buy]
			print(row)
			buy_coin(total_holding_df,row,symbol_to_buy)
			coin_held_check = True
			# print('bought {}'.format(symbol_to_buy))
			price_buy += float(total_holding_df[total_holding_df['symbol']==symbol_to_buy]['price'])
			return coin_held_check

		else:
			return print('nothing to buy right now')


	if coin_held_check == True:

		holdings = acc_check(api_key,secret_key)
		total_holding_df = total_acc_value(holdings)

		#************************
		for i in range(len(total_holding_df)):
			total_holding_df.loc[i,'value'] = float(total_holding_df.loc[i,'amount']) * float(total_holding_df.loc[i,'price'])

		symbol_to_buy = total_holding_df[total_holding_df['value']==total_holding_df['value'].max()]['symbol'].values[0]
		print(symbol_to_buy)
		#************************




		score = float(return_df[return_df['symbol'] == symbol_to_buy]['score'].values)
		current_price = float(return_df[return_df['symbol'] == symbol_to_buy]['price'].values)

		if score > .35:
			print('not selling ',symbol_to_buy,' right now')
			coin_held_check = True
			return coin_held_check 
		#####      WHAT IF TO KEEP PRICE, I WRITE INTO A CSV EVERY TIME I PURCHASE AND READ THAT FOR MOST RECENT ROW?
		elif score > .15 and score < .35 and (price_buy * 1.05 < current_price):
			#NEED TO FIND A WAY TO GET SYMBOL_TO_BUY (AND PRICE) STORED OUTSIDE OF MAIN FUNCTION SO IT CAN CONSTANTLY CHECK WHETHER IT SHOULD BE SOLD
			#MAYBE COULD DO BY GRABBING HOLDINGS - SELECTING FOR COIN WHERE PRICE X VALUE > 50
			row = return_df[return_df['symbol'] == symbol_to_buy]
			sell_coin(total_holding_df,row,symbol_to_buy)
			print('sold {}'.format(symbol_to_buy))
			symbol_to_buy = ''
			coin_held_check = False
			return coin_held_check

		elif score > .15 and score < .35 and (price_buy * 1.05 >= current_price):
			print('not selling ',symbol_to_buy,' right now')
			coin_held_check = True 
			return coin_held_check
		else:
			row = return_df[return_df['symbol'] == symbol_to_buy]
			sell_coin(total_holding_df,row,symbol_to_buy)
			print('sold {}'.format(symbol_to_buy))
			symbol_to_buy = ''
			coin_held_check = False
			return coin_held_check





	## MAY NEED TO ADD OTHER CHECKS / BALANCES

def buy_coin(total_holding_df,row,new_coin):
	# execute buy (post request)
	#once we have a coin where there is actionable item, need to identify the row in a var and pass into buy/sell funcs
	dollars = total_holding_df[total_holding_df['symbol']=='USDUSD']['amount']
	timestamp = int(time.time()*1000)
	secret_key = 'g5Ba5xMsYJoNRQMuBv4XnrbkBQ6tadRwza45e17EryvquLulIZJG30sqWHB6w7Ae'
	new_coin += str(row['symbol'])
	print(dollars.values[0])


	try:	
		q = ((float(dollars.values[0]))/float(row['price'].values[0]))*.9
		print(q)
		order = client.order_market_buy(
			symbol=row['symbol'].values[0],
			quantity=(round(q,6)))


		print('bought {} of {}'.format(((float(dollars.values[0]))/float(row['price'].values[0]))*.98,row['symbol'].values[0]))

	except Exception as a:
				print(a)
				if a.message == 'Filter failure: LOT_SIZE':
					decimal_place = 15
					while decimal_place > -1:
						try:
							q = ((float(dollars.values[0]))/float(row['price'].values[0]))*.9
							order = client.order_market_buy(
								symbol=row['symbol'].values[0],
								quantity=(round(q,decimal_place)))
							break
						except:
							decimal_place -= 1
							quantity = round(((float(dollars.values[0]))/float(row['price'].values[0]))*.9, decimal_place)
		


def sell_coin(total_holding_df,row,new_coin):
	# execute sell (post request)
	#MANDATORY API PARAMS 'symbol' (string),'side' (enum),'type' (enum)
	sell_quantity = total_holding_df[total_holding_df['symbol']==new_coin]
	timestamp = int(time.time()*1000)

	

	#JUST COPIED AND PASTED FROM BUY - FIX QUANTITY TO BE MAX AVAIL TO SELL
	
	try:	
		q = float(total_holding_df[total_holding_df['symbol']==new_coin]['amount'].values[0])*.993
		print(q)
		order = client.order_market_sell(
			symbol=row['symbol'].values[0],
			quantity=(round(q,6)))


		print('sold {} of {}'.format(q,row['symbol'].values[0]))

	except Exception as a:
				print(a)
				if a.message == 'Filter failure: LOT_SIZE':
					decimal_place = 15
					while decimal_place > -1:
						try:
							q = float(total_holding_df[total_holding_df['symbol']==new_coin]['amount'].values[0])*.993
							order = client.order_market_sell(
								symbol=row['symbol'].values[0],
								quantity=(round(q,decimal_place)))
							break
						except:
							q = float(total_holding_df[total_holding_df['symbol']==new_coin]['amount'].values[0])*.993
							decimal_place -= 1
							quantity = (round(q, decimal_place))

	new_coin = ''
	dollars = total_holding_df[total_holding_df['symbol']=='USDUSD']['amount']


def main(symbols,coin_held_check,price_buy):
	final_df = pd.DataFrame()
	list_of_klines = []
	list_of_klines_1h = []

	for i in symbols:

	
		buy_sell = '/api/v3/order'
		base = 'https://api.binance.us'

		test_connection = '/api/v3/ping'
		bids = '/api/v3/depth'
		recent_trades = '/api/v3/trades'


	
		curr_time = datetime.utcnow()
		min_ago = curr_time - timedelta(minutes=1)
		ts = datetime.timestamp(min_ago)
		time1 = datetime.utcnow()
		r_bids = requests.get('{}{}'.format(base,bids),params=dict(symbol=i))
		r_recent_trades = requests.get('{}{}'.format(base,recent_trades),params=dict(symbol=i))
		#r_15_min_trades = requests.get('{}{}'.format(base,recent_trades),params=dict(symbol=i))
		results_bids = r_bids.json()
		recent_trades = r_recent_trades.json()

		params = {
			'symbol': i,
			'interval': '1m'
			# 'start_time': ts --- not sure how we can work start_time in w/o issues with passing unneceessary params
		}
		one_min_vol = requests.get('{}{}'.format(base,klines),params=params)
		one_min_vol_data = one_min_vol.json()[0]

		params_1h = {
			'symbol': i,
			'interval': '1h'
		}
		one_hour_vol = requests.get('{}{}'.format(base,klines),params=params_1h)
		one_hour_vol_data = one_hour_vol.json()[0]
		''' 
		KLINE FORMAT
		[
		[
		1499040000000,      // Open time
		"0.01634790",       // Open
		"0.80000000",       // High
		"0.01575800",       // Low
		"0.01577100",       // Close
		"148976.11427815",  // Volume
		1499644799999,      // Close time
		"2434.19055334",    // Quote asset volume
		308,                // Number of trades
		"1756.87402397",    // Taker buy base asset volume
		"28.46694368",      // Taker buy quote asset volume
		"17928899.62484339" // Ignore.
		]
		]
		'''
		klines_1m = parse_kline_data(one_min_vol_data)
		list_of_klines.append(klines_1m)

		klines_1h = parse_kline_data(one_hour_vol_data)
		list_of_klines_1h .append(klines_1h)

		frames = {side: pd.DataFrame(data=results_bids[side], columns=['price','quantity'],dtype=float) for side in ['bids','asks']}



		df2 = pd.DataFrame(recent_trades)
		df3 = pd.DataFrame(recent_trades)

		frames_list = [frames[side].assign(side=side) for side in frames]
		data=pd.concat(frames_list,axis='index',ignore_index=True,sort=True)
		df2 = df2.sort_values('time',ascending=False)
		df3 = df3.sort_values('time',ascending=False)

		df = clean_bid_asks(data)
		##RETURN DF AFTER, RUN HISTORY PLUS STEEPNESS AFTER W KLINE_DATA_DF INCLUDED

		# kline_data_df = pd.DataFrame(list_of_klines)
		run = history_plus_steepness(df,curr_time,df2,df3)
		run['symbol'] = i


		temp_df = pd.DataFrame([run])
		final_df = final_df.append(temp_df,ignore_index=True)


		#initialize account value. set var to half of it
		#connect to api for orders
		#some combo of scores for check1 and check2 to maximize (similar formula to f1 score)?
		#for i in final_df choose max(f1 score)
		#buy symbol with half acc value
		#while buy open - stop searching for buys
		#if verdict = sell for over one min consecutively - sell
		#start searching for new buys again

	holdings = acc_check(api_key,secret_key)
	total_holding_df = total_acc_value(holdings)
	half_acc = acc_value_final(total_holding_df)

	n = half_acc/10
	print(total_holding_df)

	kline_data_df_1m = pd.DataFrame(list_of_klines)
	kline_data_df_1h = pd.DataFrame(list_of_klines_1h)

	vol_spike_check = comp_kline_time_diff(kline_data_df_1m,kline_data_df_1h)

	df = final_df.merge(total_holding_df,on='symbol')
	df = df.join(kline_data_df_1m,how='outer')

	return_df = vol_spike_override(df)
	return_df = return_df.drop(['taker_buy_base_asset_vol','taker_buy_quote_asset_vol','ignore','open_time','close','quote_asset_volume','close_time','high','low','open','num_trades'],axis=1)

	# print(return_df)
	# account_management = manage_holdings(return_df)
	# print(account_management)
	coin_held_check = manage_holdings(return_df,coin_held_check,price_buy)
	print('coin held check: ',coin_held_check)

	new_coin = ''
	return coin_held_check



if __name__ == '__main__':
	coin_held_check = False
	price_buy = 0
	while True:
		coin_held_check = main(symbols,coin_held_check,price_buy)
		print('coin held check: ',coin_held_check)


####          NOTES TO DO

####		1. Tighten margins for buy/sell quantity (ideally want to sell everything available)



####    SWITCH FROM F_SCORE TO SIGMOID FUNCTIONS ACROSS ALL??? WHAT WOULD BE IDEAL SCORE RANGES??
'''   
	S(X) = 1 / 1+ e^(-x)
	s(1000) = 1/1+e^-1000 = 1
	S(0) = 1/1+e^(0) = .5
	S(1) = 1/1+e^(-1) = .73
	S(2) = 1/1+e^(-2) = .88
	S(3) ~ .95
	S(10) ~ 1
	S(5) ~ .993
	-------> this is GREAT for bid/ask. volume will have more variance so may not be appropriate. Another 0-1 func ideal


	hyperbolic tangent function (tanh) f(z) = tanh(z) = 2O(2z)-1 = 2(1/1+e^-2z)-1
	---> after values of 2 basically returns 1. would be good buy_to_sell_15

	tanh(.1) = .1
	tanh(.25) = .245
	tanh(.5) = .46
	tanh(1) = .76
	tanh(2) = .96
	tanh(3) = .995
	tanh(5) = .9999

	want smoother func for volume_5m
	maybe tanh(x/3)?? much smoother



'''





