import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import csv
from sklearn.model_selection import train_test_split,cross_val_score,KFold 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report,confusion_matrix
import sys



final_df = pd.read_csv('binance_data_final_edit_1.csv')



X = final_df[['bid_price','bid_quantity','ask_price','ask_quantity','btc_price','btc_quantity','btc_trade_val','market_buy_sell','trade_price','trade_quantity']].values
y = final_df['BSH'].values

enc = OneHotEncoder()

y = enc.fit_transform(y[:,np.newaxis]).toarray()

# print(enc.get_feature_names())


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=.3)


'''trying kfold'''
--> should try to shuffle data, maybe use less folds. almost certainly need more data

inputs = np.concatenate((X_train,X_test), axis=0)
targets = np.concatenate((y_train,y_test), axis=0)

kfold = KFold(n_splits=5,shuffle=True)

fold_no = 1 
acc_per_fold = []
loss_per_fold = []
for train,test in kfold.split(inputs,targets):
	model = Sequential()
	model.add(Dense(100,activation='relu'))
	model.add(Dropout(rate=.1))
	model.add(Dense(50,activation='relu'))
	model.add(Dropout(rate=.1))
	model.add(Dense(25,activation='relu'))
	model.add(Dropout(rate=.5))
	model.add(Dense(3,activation='softmax'))

	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

	print('---------------------------')
	print('Training for fold {}'.format(fold_no))

	history = model.fit(inputs[train],targets[train],batch_size=10,epochs=100,verbose=1)

	scores = model.evaluate(inputs[test],targets[test],verbose=0)
	print('Score for fold {}:'.format(fold_no),model.metrics_names[0],'of {}; {} of {}%'.format(scores[0],model.metrics_names[1],scores[1]*100))
	acc_per_fold.append(scores[1]*100)
	loss_per_fold.append(scores[0])

	fold_no = fold_no + 1

	

sys.exit()
n_features = X.shape[1]
n_classes = y.shape[1]

model = Sequential()

model.add(Dense(100,activation='relu'))
model.add(Dropout(rate=.1))
model.add(Dense(50,activation='relu'))
model.add(Dropout(rate=.1))
model.add(Dense(25,activation='relu'))
model.add(Dropout(rate=.5))
model.add(Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(X_train,y_train,batch_size=10,epochs=50,validation_data=(X_test,y_test))
print(history)
preds = model.predict(X_test)
# print(preds)

reading = (preds>.5).astype(np.int)
print(enc.get_feature_names)
print(reading)

print(classification_report(y_test,reading))

score = model.evaluate(X_test,y_test)
print(score)
print('Test loss:',score[0])
print('Test accuracy:',score[1])



# def clean_BSH(row):

# 	if row == 'Buy':
# 		row = 2
# 	elif row == 'Sell':
# 		row = 0
# 	else:
# 		row = 1

# 	return row

# final_df['BSH'] = final_df['BSH'].apply(lambda x: clean_BSH(x))

# X = final_df[['bid_price','bid_quantity','ask_price','ask_quantity','btc_price','btc_quantity','btc_trade_val','market_buy_sell','trade_price','trade_quantity']].values
# y = final_df['BSH'].values

# # encoder = LabelEncoder()
# # encoder.fit(final_df['BSH'])
# # encoded_Y = encoder.transform(Y)

# # dummy_y = np_utils.to_categorical(encoded_Y)

# print(final_df)

# # X = final_df.drop(['BSH','datetime'],axis=1).values



# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.25)
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# early_stop = EarlyStopping(monitor='val_loss',mode='min',patience=10)

# model = Sequential()

# model.add(Dense(25,activation='relu'))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(3,activation='softmax'))

# model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# model.fit(X_train,y_train,epochs=100,validation_data=(X_test,y_test))

# preds = model.predict_classes(X_test)

# print(classification_report(y_test,preds))
# print(confusion_matrix(y_test,preds))


# def baseline_model():
# 	model = Sequential()

# 	model.add(Dense(25,activation='relu'))
# 	model.add(Dense(10,activation='relu'))
# 	model.add(Dense(3,activation='softmax'))

# 	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# 	return model 

# KERAS_PARAMS = dict(epochs=100,batch_size=10)

# clf = LabelPowerset(classifier=Keras(baseline_model,True,KERAS_PARAMS,require_dense=[True,True]))
# clf.fit(X_train,y_train)
# y_pred = clf.predict(X_test)


# estimator = KerasClassifier(build_fn=baseline_model,epochs=100,batch_size=10)

# kfold = KFold(n_splits=5, shuffle=True)

# results = cross_val_score(estimator,X,dummy_y,cv=kfold)

# print('result:',results.mean()*100, results.std()*100)

	# model.fit(X_train,y_train,epochs=100,validation_data=(X_test,y_test))



# plt.scatter(df.index,df['trade_price'])
# plt.show()

# fig,ax = plt.subplots()
# ax.scatter(df.index,df['trade_price'])

# for i, txt in enumerate(df.index):
# 	ax.annotate(txt,(df.index[i],df['trade_price'][i]))

# plt.show()