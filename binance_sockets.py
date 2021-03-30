import asyncio
import websocket
from binance.client import Client 
from binance.websockets import BinanceSocketManager
import time
import json
import sys
sys.path.append('../')
import pandas as pd
from datetime import datetime
from pytz import timezone
import pytz
import numpy as np
import csv







api_key = '{key}'
secret_key = '{key}'
client = Client(api_key='{}'.format(api_key),api_secret='{}'.format(secret_key),tld='us')



base='wss://stream.binance.com:9443/ws/'
#raw streams
stream='bnbusd@depth'

bm = BinanceSocketManager(client)
sm = BinanceSocketManager(client)
tm = BinanceSocketManager(client)
btc = BinanceSocketManager(client)





# df = pd.DataFrame(columns=('price','quantity'))
def process_bids(msg):
	'''
	We are placing each bid and each ask into a tuple, then creating a list of those tuples, then using stack_bids_asks to enter them into a dataframe
	we are going to want to create a dataframe for each %set period of time% in order to have data we can consistently analyze 
	'''

	
	# bids = pd.DataFrame(columns=['minute','bid_price','bid_quantity'])
	# asks = pd.DataFrame(columns=['minute','ask_price','ask_quantity'])
	# list_of_tups = []

	# - appears we cant run both simultaneously? might need an async type function

	file_to_output_update = open('init_binance_data_bids_hbar.csv',mode='a',newline='')
	csv_writer_bids = csv.writer(file_to_output_update,delimiter=',')
	# csv_writer_bids.writerow(['datetime','bid_price','bid_quantity'])

	#***might need to initialize file outside of process_msg function and then use mode='a' from there on to append rather then reinitialize every time socket connection runs


	if len(msg['b']) == 0:
		pass
	else:
		for i in range(len(msg['b'])):
			if float(msg['b'][i][1]) == 0:
				pass
			else:
				utc = datetime.utcfromtimestamp(msg['E']/1000)
				bids = pd.DataFrame([[utc,msg['b'][i][0],msg['b'][i][1]]],columns=['datetime','bid_price','bid_quantity'])
				

				print(bids)
				# with open('hbar_price_yada_yada2.csv','w') as myfilehbar:
				# wrtr = csv.writer(myfilehbar,delimiter=',')
				# for i in bids:
				# 	csv_writer_bids.writerow(bids[i])

				rows = zip(bids['datetime'],bids['bid_price'],bids['bid_quantity'])
				for row in rows:
					csv_writer_bids.writerow(row)
				
				#csv_writer_bids.writerow([bids['datetime'],bids['bid_price'],bids['bid_quantity']])
				# csv_writer_bids.writerow(bid)
				# 	print(bid)




	

def process_asks(msg):

		file_to_output_update_asks = open('init_binance_data_asks_hbar.csv',mode='a',newline='')
		csv_writer_asks = csv.writer(file_to_output_update_asks,delimiter=',')

		if len(msg['a']) == 0:
			pass
		else:
			for i in range(len(msg['a'])):
				if float(msg['a'][i][1]) == 0:
					pass
				else:
					utc = datetime.utcfromtimestamp(msg['E']/1000)
					asks = pd.DataFrame([[utc,msg['a'][i][0],msg['a'][i][1]]],columns=['datetime','ask_price','ask_quantity'])
					print(asks)

					rows = zip(asks['datetime'],asks['ask_price'],asks['ask_quantity'])
					for row in rows:
						csv_writer_asks.writerow(row)


#trades may need to be stored elsewhere unless we are grouping by minute 

def process_trades(msg):

	# print(msg)
	#--> maybe make loop - while curr_min == True, count trade value of buys and trade value of sells, else - reset current minute and begin aggregating again. can write on by min basis to csv?
	file_to_output_trades_update = open('init_binance_data_trades_hbar.csv',mode='a',newline='')
	csv_writer_trades_update = csv.writer(file_to_output_trades_update,delimiter=',')

	print(msg)
	buys = pd.DataFrame(columns=['buys','datetime','price_trade','quantity_trade'])
	sells = pd.DataFrame(columns=['sells','datetime','price_trade','quantity_trade'])
	if msg['m'] == False:
		utc = datetime.utcfromtimestamp(msg['E']/1000)
		df_buy_row = pd.DataFrame(data=np.array([[1,utc,msg['p'],msg['q']]]),columns=['buys','datetime','price_trade','quantity_trade'])
		final_buys = pd.concat([buys,df_buy_row],ignore_index=True)
		print(final_buys)
		rows = zip(df_buy_row['buys'],df_buy_row['datetime'],df_buy_row['price_trade'],df_buy_row['quantity_trade'])
		for row in rows:
			csv_writer_trades_update.writerow(row)

	if msg['m'] == True:
		utc = datetime.utcfromtimestamp(msg['E']/1000)
		df_sells_row = pd.DataFrame(data=np.array([[0,utc,msg['p'],msg['q']]]),columns=['sells','datetime','price_trade','quantity_trade'])
		final_sells = pd.concat([sells,df_sells_row],ignore_index=True)
		print(final_sells)
		rows = zip(df_sells_row['sells'],df_sells_row['datetime'],df_sells_row['price_trade'],df_sells_row['quantity_trade'])
		for row in rows:
			csv_writer_trades_update.writerow(row)

#sells will return 0,  buys return 1






def compare_price(a,b):
	if a == b:
		print('same')
	elif a > b:
		print('up')
	else:
		print('down')

def bitcoin_price(msg):
	''' want btc price at the start of each minute and btc volume from prior minute. also want btc direction of price movement from prior min to start of current min
	'''
	file_to_output_btc_update = open('init_binance_data_btc.csv',mode='a',newline='')
	csv_writer_btc_update = csv.writer(file_to_output_btc_update,delimiter=',')


	utc = datetime.utcfromtimestamp(msg['E']/1000)
	# print('current time:',utc)
	# print('bitcoin price:',msg['c']) #bitcoin most recent price
	# print('bitcoin quantity:', msg['Q']) #last quantity
	total_val = float(msg['c']) * float(msg['Q']) #most recent trade value
	# print('most recent bitcoin trade value',total_val)
	btc_df = pd.DataFrame(data=np.array([[utc,msg['c'],msg['Q'],total_val]]),columns=['current_time','bitcoin_price','bitcoin_quantity','recent_trade_val'])
	print(btc_df)
	rows = zip(btc_df['current_time'],btc_df['bitcoin_price'],btc_df['bitcoin_quantity'],btc_df['recent_trade_val'])
	for row in rows:
		csv_writer_btc_update.writerow(row)
	# return btc_df
	




file_to_output_bids = open('init_binance_data_bids_hbar.csv',mode='w',newline='')
csv_writer_bids = csv.writer(file_to_output_bids,delimiter=',')
csv_writer_bids.writerow(['datetime','bid_price','bid_quantity'])

bids = bm.start_depth_socket('HBARUSDT',process_bids)


file_to_output_asks = open('init_binance_data_asks_hbar.csv',mode='w',newline='')
csv_writer_asks = csv.writer(file_to_output_asks,delimiter=',')
csv_writer_asks.writerow(['datetime','ask_price','ask_quantity'])


asks = sm.start_depth_socket('HBARUSDT',process_asks)


file_to_output_trades = open('init_binance_data_trades_hbar.csv',mode='w',newline='')
csv_writer_trades = csv.writer(file_to_output_trades,delimiter=',')
csv_writer_trades.writerow(['buy_sell','datetime','price_trade','quantity_trade'])


trades = tm.start_trade_socket('HBARUSDT',process_trades)


file_to_output_btc = open('init_binance_data_btc.csv',mode='w',newline='')
csv_writer_btc = csv.writer(file_to_output_btc,delimiter=',')
csv_writer_btc.writerow(['datetime','bitcoin_price','bitcoin_quantity','trade_val'])

btc_price = btc.start_symbol_ticker_socket('BTCUSDT',bitcoin_price)



bm.start()


sm.start()


tm.start()


btc.start()


# time.sleep(30)

# btc.close()
# print(final_df)





