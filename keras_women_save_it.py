from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.feature_extraction import DictVectorizer
import numpy
import pandas as pd
from pandas import DataFrame
from sklearn.utils import shuffle
import numpy as np
from functools import *
from itertools import chain
from keras import optimizers
# fix random seed for reproducibility
numpy.random.seed(0)
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, recall_score, auc

earlystop = EarlyStopping(monitor='val_acc',
							  mode='auto'
							, patience=5
							  )

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
								  patience=5, min_lr=0.001)

callbacks_list = [
		earlystop
		# ,
		# reduce_lr
	]

scaler = None

def make_prediction(df, model):
	try:
		loaded_model = model
		print('inside make prediction')
		### now predict label for the data ###
		X_test = df.value
		X_test_s = scaler.transform(X_test)

		p = loaded_model.predict(X_test_s)
		return p
	except:
		print('Error in making prediction')
		return None

def make_model(df):
	key = 'is_female'
	df[key] = df[key].astype(str)

	y = df['is_female']
	df = df.drop('is_female', axis = 1)
	# create model
	model = Sequential()
	model.add(Dense(500, input_dim=df.shape[1], activation='relu'))
	model.add(Dense(200, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(200, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(200, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(200, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(80, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	# Compile model
	# opt = optimizers.Adam(lr=0.005, decay=0.2)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	X_train  = df.values

	scaler = StandardScaler().fit(X_train)
	X_train_s = scaler.transform(X_train)

	X_train, X_test, y_train, y_test = train_test_split(X_train_s, y, test_size=0.15, random_state=0)

	# model.fit(X_train, y_train, epochs=1)
	model.fit(X_train, y_train, epochs=200,
			  batch_size=128*2,
			  validation_split=0.10,
			  verbose=2,
			  shuffle=True
			  )

	scores = model.evaluate(X_test, y_test)
	print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))



	y_pred = model.predict(X_test)
	print "PRED "
	print y_pred[10:25]

	# y_rounded = []
	# for x in y_pred:
	# 	val = x[0]
	# 	if val <= 0.9:
	# 		y_rounded.append('0')
	# 	else:
	# 		y_rounded.append('1')
	#
	# print "TEST "
	# print y_test[10:25].values
	#
	# # y_rounded = [round(x[0]) for x in y_pred]
	# # print "Rounded "
	# # print y_rounded[10:25]
	#
	# print confusion_matrix(y_test, y_rounded)
	# try:
	# 	model_recall_score = recall_score(y_test, y_rounded, average='macro')
	# 	auc_score = roc_auc_score(y_test, y_rounded)
	# 	print("recall score", model_recall_score, " auc score ", auc_score)
	# except:
	# 	print("Error calculating recall score")

	model.save('_select_25_02.h5')


	return model

#  _ big_4.csv is gold, keep it. do not replace/ rewrite
def get_data(name):
	file = name + '_big_4.csv'
	df = pd.read_csv(file)
	print(name + '  file read ', df.shape)
	df.fillna('NA', inplace=True)
	# print('To delete = ',to_delete)
	return df


if __name__ == "__main__":

	df = get_data('train')

	model = make_model(df)
	print("df shape ", df.shape)

	df_t = get_data('test')
	label = make_prediction(df_t, model)
	# print("label=", label.tolist())
	scores = list(chain.from_iterable(label.tolist()))

	print("df test shape ", df_t.shape)
	output = []
	for i in scores:
		if i <= 0.5:
			output.append(0)
		else:
			output.append(1)

	index = range(0, len(output))
	print(output[:20])
	print(len(index))

	df = DataFrame({'test_id': index, 'is_female': output})
	print(df.head())
	df.to_csv(path_or_buf='submission_4_keras_adam_save_it_feb_25.csv', index=False)
